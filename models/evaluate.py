"""models.evaluate — evaluate a trained BFId LSTM on its held-out test split.

CLI::

    python3 -m models.evaluate --config experiments/bfi_identity_baseline.yaml
    python3 -m models.evaluate --model data/models/bfi_identity_baseline

Loads the model + fitted Standardiser + label vocab + saved split written by
:mod:`models.train`, then reports accuracy overall and broken down *by
participant*, *by style* and *by perspective* — matching the paper's evaluation
breakdowns (Todt, Morsbach, Strufe, CCS '25, sec. 5.3). Also reports top-2
accuracy (the paper's empty-room control metric).

Writes ``<data_root>/reports/<name>.json`` plus a short ``.txt`` summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from wallflower import contract
from models import metrics
from models.dataset import (
    BFIdDataset,
    SampleRow,
    collate_fn,
    load_index,
    stratified_split,
)
from models.dataset import Standardiser
from models.lstm_classifier import BFIdLSTM
from models.train import TrainConfig, config_from_yaml


# --------------------------------------------------------------------------- #
# Artefact loading
# --------------------------------------------------------------------------- #
def _row_from_npz_path(p: Path) -> SampleRow:
    """Reconstruct a SampleRow from a sample npz path (canonical basename)."""
    parts = Path(p).stem.split("_")
    participant, style, _trial, modality, ptag = parts[:5]
    perspective = int(ptag[1:])
    return SampleRow(Path(p), participant, style, perspective, modality)


def load_artifacts(model_dir: Path) -> dict:
    model_dir = Path(model_dir)
    ckpt = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=False)
    model = BFIdLSTM.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    standardiser = Standardiser.load(model_dir / "standardiser.json")
    label_to_idx = json.loads((model_dir / "label_vocab.json").read_text())
    split = json.loads((model_dir / "split.json").read_text())
    return {
        "model": model,
        "standardiser": standardiser,
        "label_to_idx": {k: int(v) for k, v in label_to_idx.items()},
        "split": split,
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_model(
    model: BFIdLSTM,
    rows: list[SampleRow],
    label_to_idx: dict[str, int],
    standardiser: Standardiser,
    modality: str,
    *,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict:
    """Run inference over `rows`; return overall + grouped accuracy report."""
    dev = torch.device(device)
    model = model.to(dev)
    ds = BFIdDataset(
        rows, modality=modality, label_to_idx=label_to_idx, standardiser=standardiser
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    y_true: list[int] = []
    y_pred: list[int] = []
    all_logits: list[np.ndarray] = []
    styles: list[str] = []
    perspectives: list[int] = []
    participants: list[str] = []

    model.eval()
    with torch.no_grad():
        for x, lengths, labels, metas in loader:
            x = x.to(dev)
            logits = model(x, lengths)
            all_logits.append(logits.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())
            styles.extend(m["style"] for m in metas)
            perspectives.extend(int(m["perspective"]) for m in metas)
            participants.extend(m["participant"] for m in metas)

    logits_mat = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 1))
    num_classes = len(label_to_idx)

    report: dict[str, Any] = {
        "n_test": len(y_true),
        "num_classes": num_classes,
        "overall": {
            "accuracy": metrics.accuracy(y_true, y_pred),
            "top2_accuracy": metrics.top_k_accuracy(y_true, logits_mat, k=2),
        },
        "by_participant": metrics.per_group_accuracy(y_true, y_pred, participants),
        "by_style": metrics.per_group_accuracy(y_true, y_pred, styles),
        "by_perspective": {
            str(k): v
            for k, v in metrics.per_group_accuracy(y_true, y_pred, perspectives).items()
        },
        "confusion_matrix": metrics.confusion_matrix(
            y_true, y_pred, num_classes
        ).tolist(),
        "labels": [lbl for lbl, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])],
    }
    return report


def _resolve_test_rows(cfg: TrainConfig, split: dict) -> list[SampleRow]:
    """Use the saved test split if it has rows; else re-derive from the index."""
    test_paths = split.get("test") or []
    rows = [_row_from_npz_path(Path(p)) for p in test_paths if Path(p).exists()]
    if rows:
        return rows
    # Fallback: rebuild from index with the same seed/frac (e.g. a tiny dataset
    # where the test split was empty and train was reused).
    all_rows = load_index(
        cfg.data_root, modality=cfg.modality, styles=cfg.styles,
        perspectives=cfg.perspectives, participants=cfg.participants,
    )
    s = stratified_split(
        all_rows, test_frac=split.get("test_frac", cfg.test_frac),
        seed=split.get("seed", cfg.split_seed),
    )
    return s.test or s.train


def _format_summary(name: str, modality: str, report: dict) -> str:
    lines = [
        "BFId baseline evaluation.",
        f"experiment : {name}",
        f"modality   : {modality}",
        f"test size  : {report['n_test']}  classes: {report['num_classes']}",
        f"accuracy   : {report['overall']['accuracy'] * 100:.2f}%",
        f"top-2 acc  : {report['overall']['top2_accuracy'] * 100:.2f}%",
        "",
        "by style:",
    ]
    for k, v in report["by_style"].items():
        lines.append(f"  {k:<12} {v['accuracy'] * 100:6.2f}%  (n={v['n']})")
    lines.append("by perspective:")
    for k, v in report["by_perspective"].items():
        lines.append(f"  p{k:<11} {v['accuracy'] * 100:6.2f}%  (n={v['n']})")
    lines.append("by participant:")
    for k, v in report["by_participant"].items():
        lines.append(f"  {k:<12} {v['accuracy'] * 100:6.2f}%  (n={v['n']})")
    return "\n".join(lines) + "\n"


def run_evaluation(cfg: TrainConfig, model_dir: Path) -> dict:
    if cfg.modality not in contract.MODALITIES:
        raise SystemExit(
            f"[evaluate] unknown modality {cfg.modality!r}; "
            f"expected one of {contract.MODALITIES}"
        )
    art = load_artifacts(model_dir)
    rows = _resolve_test_rows(cfg, art["split"])
    if not rows:
        raise SystemExit("[evaluate] no test samples available to evaluate")

    report = evaluate_model(
        art["model"], rows, art["label_to_idx"], art["standardiser"],
        modality=cfg.modality, batch_size=cfg.batch_size,
    )
    report["name"] = cfg.name
    report["modality"] = cfg.modality
    report["research_only"] = True
    report["model_dir"] = str(model_dir)

    reports_dir = Path(cfg.data_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{cfg.name}.json").write_text(json.dumps(report, indent=2))
    summary = _format_summary(cfg.name, cfg.modality, report)
    (reports_dir / f"{cfg.name}.txt").write_text(summary)
    print(summary)
    print(f"[evaluate] report -> {reports_dir / (cfg.name + '.json')}")
    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained BFId LSTM identity classifier "
                    "(reports paper-style breakdowns)."
    )
    p.add_argument("--config", type=Path, default=None,
                   help="experiment yaml (same one used for training)")
    p.add_argument("--model", type=Path, default=None,
                   help="trained model dir (default data/models/<name>)")
    p.add_argument("--data-root", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.config is not None:
        cfg = config_from_yaml(args.config)
    else:
        cfg = TrainConfig()
    if args.data_root is not None:
        cfg.data_root = args.data_root

    model_dir = args.model if args.model is not None else cfg.resolved_out_dir()
    run_evaluation(cfg, model_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
