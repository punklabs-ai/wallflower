"""models.train — train the baseline BFId LSTM identity classifier.

CLI::

    python3 -m models.train --config experiments/bfi_identity_baseline.yaml
    python3 -m models.train --modality bfi --epochs 20 --lr 1e-3 --batch-size 16

Implements the paper's deliberately simple training loop (Todt, Morsbach,
Strufe, CCS '25, sec. 5.2): standardise features (train-fit only) -> LSTM
-> FC/BN/ReLU/FC -> CrossEntropy, optimised with Adam, default 80/20 split.
CPU-friendly defaults so a small dataset trains in seconds on node1.

Artefacts written to ``<out_dir>`` (default ``data/models/<name>``):
  * ``model.pt``         torch state_dict + model config,
  * ``standardiser.json`` fitted (train-only) Standardiser stats,
  * ``label_vocab.json`` participant ``label <-> index`` map,
  * ``split.json``       the exact train/test npz lists (for evaluate.py),
  * ``run_report.json``  per-epoch loss/acc + final summary.

An optional Optuna hook (``optuna.enabled``) is provided but NOT required.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from wallflower import contract
from models import metrics
from models.dataset import (
    BFIdDataset,
    collate_fn,
    load_index,
    stratified_split,
)
from models.lstm_classifier import BFIdLSTM


# --------------------------------------------------------------------------- #
# Config resolution (yaml + flag overrides)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    name: str = "bfi_identity_baseline"
    modality: str = "bfi"
    data_root: str = "data"
    styles: list[str] | None = None
    perspectives: list[int] | None = None
    participants: list[str] | None = None
    test_frac: float = 0.2
    split_seed: int = 1337
    hidden: int = 128
    lstm_layers: int = 2
    fc_hidden: int = 64
    dropout: float = 0.2
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 16
    weight_decay: float = 0.0
    device: str = "auto"
    train_seed: int = 1337
    out_dir: str | None = None
    repeats: int = 1
    optuna_enabled: bool = False
    optuna_trials: int = 20

    extra: dict = field(default_factory=dict)

    def resolved_out_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir)
        return Path(self.data_root) / "models" / self.name


def _load_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def config_from_yaml(path: Path) -> TrainConfig:
    """Build a TrainConfig from an experiment yaml.

    Accepts the canonical experiment schema (``data.*`` / ``optim.*`` /
    ``split.train_frac``) used by experiments/bfi_identity_baseline.yaml, and
    also tolerates a flat ``filter.*`` / ``train.*`` / ``split.test_frac``
    style so older configs keep working.
    """
    raw = _load_yaml(path)
    # data filters: prefer `data:` block, fall back to `filter:`.
    data = raw.get("data") or raw.get("filter") or {}
    split = raw.get("split", {}) or {}
    model = raw.get("model", {}) or {}
    # optim: prefer `optim:` block, fall back to `train:`.
    optim = raw.get("optim") or raw.get("train") or {}
    opt = raw.get("optuna", {}) or {}

    # split fraction: train_frac (canonical) or test_frac (flat).
    if "train_frac" in split:
        test_frac = round(1.0 - float(split["train_frac"]), 10)
    else:
        test_frac = split.get("test_frac", 0.2)

    return TrainConfig(
        name=raw.get("name", "bfi_identity_baseline"),
        modality=raw.get("modality", "bfi"),
        data_root=raw.get("data_root", "data"),
        styles=data.get("styles"),
        perspectives=data.get("perspectives"),
        participants=data.get("participants"),
        test_frac=test_frac,
        split_seed=split.get("seed", 1337),
        hidden=model.get("hidden", 128),
        lstm_layers=model.get("lstm_layers", 2),
        fc_hidden=model.get("fc_hidden", 64),
        dropout=model.get("dropout", 0.2),
        epochs=optim.get("epochs", 20),
        lr=optim.get("lr", 1e-3),
        batch_size=optim.get("batch_size", 16),
        weight_decay=optim.get("weight_decay", 0.0),
        device=optim.get("device", raw.get("device", "auto")),
        train_seed=split.get("seed", 1337),
        out_dir=raw.get("out_dir"),
        repeats=split.get("repeats", 1),
        optuna_enabled=opt.get("enabled", False),
        optuna_trials=opt.get("n_trials", 20),
    )


def _apply_flag_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.modality is not None:
        cfg.modality = args.modality
    if args.data_root is not None:
        cfg.data_root = args.data_root
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.hidden is not None:
        cfg.hidden = args.hidden
    if args.lstm_layers is not None:
        cfg.lstm_layers = args.lstm_layers
    if args.fc_hidden is not None:
        cfg.fc_hidden = args.fc_hidden
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.train_seed = args.seed
        cfg.split_seed = args.seed
    if args.name is not None:
        cfg.name = args.name
    if args.out is not None:
        cfg.out_dir = args.out
    return cfg


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _eval_loader(model: nn.Module, loader: DataLoader,
                 device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, lengths, labels, _meta in loader:
            x = x.to(device)
            logits = model(x, lengths)
            preds = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())
    return y_true, y_pred


# --------------------------------------------------------------------------- #
# Core training routine
# --------------------------------------------------------------------------- #
def train_once(cfg: TrainConfig, *, verbose: bool = True) -> dict:
    """Build dataset+split, train, save artefacts. Returns the run report dict."""
    seed_everything(cfg.train_seed)
    device = resolve_device(cfg.device)

    rows = load_index(
        cfg.data_root,
        modality=cfg.modality,
        styles=cfg.styles,
        perspectives=cfg.perspectives,
        participants=cfg.participants,
    )
    if not rows:
        raise SystemExit(
            f"[train] no samples found under {cfg.data_root}/processed for "
            f"modality={cfg.modality} styles={cfg.styles}. Run the parsers first."
        )

    split = stratified_split(rows, test_frac=cfg.test_frac, seed=cfg.split_seed)
    if verbose:
        print(f"[train] {len(rows)} samples -> {len(split.train)} train / "
              f"{len(split.test)} test (modality={cfg.modality})")

    # Fixed label vocab over ALL rows so train/test share class indices.
    labels = sorted({r.participant for r in rows})
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    num_classes = len(label_to_idx)
    if num_classes < 2:
        raise SystemExit(
            f"[train] need >= 2 participants to train a classifier, got {num_classes}"
        )

    train_ds = BFIdDataset(
        split.train, modality=cfg.modality, label_to_idx=label_to_idx
    )
    # Standardiser fit on TRAIN ONLY (paper: no preprocessing beyond standardise).
    standardiser = train_ds.fit_standardiser()
    test_ds = BFIdDataset(
        split.test or split.train, modality=cfg.modality,
        label_to_idx=label_to_idx, standardiser=standardiser,
    )

    g = torch.Generator()
    g.manual_seed(cfg.train_seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, generator=g,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
    )

    input_dim = contract.expected_feature_dim(cfg.modality, with_dt=True)
    model = BFIdLSTM(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden=cfg.hidden,
        lstm_layers=cfg.lstm_layers,
        fc_hidden=cfg.fc_hidden,
        dropout=cfg.dropout,
    ).to(device)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    epoch_log: list[dict] = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for x, lengths, batch_labels, _meta in train_loader:
            x = x.to(device)
            batch_labels = batch_labels.to(device)
            optimiser.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimiser.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
        train_loss = running / max(n_seen, 1)

        yt, yp = _eval_loader(model, test_loader, device)
        test_acc = metrics.accuracy(yt, yp)
        rec = {"epoch": epoch, "train_loss": train_loss, "test_accuracy": test_acc}
        epoch_log.append(rec)
        if verbose:
            print(f"[train] epoch {epoch:3d}/{cfg.epochs} "
                  f"loss={train_loss:.4f} test_acc={test_acc:.4f}")

    train_secs = time.time() - t0

    # ---- persist artefacts ------------------------------------------------ #
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {"state_dict": model.state_dict(), "model_config": model.config()},
        out_dir / "model.pt",
    )
    standardiser.save(out_dir / "standardiser.json")
    (out_dir / "label_vocab.json").write_text(json.dumps(label_to_idx, indent=2))
    (out_dir / "split.json").write_text(json.dumps({
        "test_frac": cfg.test_frac,
        "seed": cfg.split_seed,
        "train": [str(r.npz_path) for r in split.train],
        "test": [str(r.npz_path) for r in split.test],
    }, indent=2))

    report: dict[str, Any] = {
        "name": cfg.name,
        "modality": cfg.modality,
        "research_only": True,
        "num_classes": num_classes,
        "labels": labels,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "input_dim": input_dim,
        "model_config": model.config(),
        "hyperparams": {
            "epochs": cfg.epochs, "lr": cfg.lr, "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay, "dropout": cfg.dropout,
        },
        "device": str(device),
        "train_seconds": round(train_secs, 3),
        "epochs": epoch_log,
        "final_test_accuracy": epoch_log[-1]["test_accuracy"] if epoch_log else None,
        "out_dir": str(out_dir),
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    if verbose:
        print(f"[train] artefacts -> {out_dir}")
    return report


# --------------------------------------------------------------------------- #
# Optional Optuna hook (NOT required to run)
# --------------------------------------------------------------------------- #
def run_optuna(cfg: TrainConfig) -> dict:
    """Optional per-modality hyperparameter search (paper tuned via Optuna).

    Imported lazily so the dependency is never required for the baseline path.
    """
    import optuna  # local import; [ml] extra

    def objective(trial: "optuna.Trial") -> float:
        trial_cfg = TrainConfig(**{**cfg.__dict__})
        trial_cfg.hidden = trial.suggest_categorical("hidden", [64, 128, 256])
        trial_cfg.lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        trial_cfg.fc_hidden = trial.suggest_categorical("fc_hidden", [32, 64, 128])
        trial_cfg.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        trial_cfg.name = f"{cfg.name}_optuna_{trial.number}"
        rep = train_once(trial_cfg, verbose=False)
        return float(rep["final_test_accuracy"] or 0.0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.optuna_trials)
    return {"best_params": study.best_params, "best_value": study.best_value}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the baseline BFId LSTM identity classifier."
    )
    p.add_argument("--config", type=Path, default=None,
                   help="experiment yaml (e.g. experiments/bfi_identity_baseline.yaml)")
    p.add_argument("--modality", choices=list(contract.MODALITIES), default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--lstm-layers", type=int, default=None)
    p.add_argument("--fc-hidden", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--out", default=None, help="output dir (default data/models/<name>)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config is not None:
        cfg = config_from_yaml(args.config)
    else:
        cfg = TrainConfig()
    cfg = _apply_flag_overrides(cfg, args)

    if cfg.optuna_enabled:
        result = run_optuna(cfg)
        print(json.dumps({"optuna": result}, indent=2))
        return 0

    if cfg.repeats and cfg.repeats > 1:
        train_repeated(cfg)
    else:
        train_once(cfg)
    return 0


def train_repeated(cfg: TrainConfig) -> dict:
    """Train over `repeats` seeded splits; report mean +- std (paper reporting).

    The base ``split_seed`` is offset per repeat. The final repeat's artefacts
    remain in ``<out_dir>``; an aggregate summary is written alongside.
    """
    accs: list[float] = []
    base_seed = cfg.split_seed
    base_name = cfg.name
    for r in range(cfg.repeats):
        rep_cfg = TrainConfig(**{**cfg.__dict__})
        rep_cfg.split_seed = base_seed + r
        rep_cfg.train_seed = base_seed + r
        rep_cfg.repeats = 1
        rep_cfg.name = base_name  # share out_dir; last repeat's model persists
        print(f"[train] repeat {r + 1}/{cfg.repeats} (seed={rep_cfg.split_seed})")
        report = train_once(rep_cfg, verbose=True)
        accs.append(float(report["final_test_accuracy"] or 0.0))

    summary = {
        "name": base_name,
        "modality": cfg.modality,
        "repeats": cfg.repeats,
        "per_split_accuracy": accs,
        "accuracy_mean_std": metrics.mean_std(accs),
        "formatted": metrics.format_mean_std(accs),
    }
    out_dir = cfg.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repeated_runs.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] repeated accuracy = {summary['formatted']}  -> {out_dir}")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
