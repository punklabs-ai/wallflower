"""models.dataset — torch Dataset over the processed BFId sample index.

Reads ``data/processed/samples.parquet`` (the per-sample index that
:mod:`parsers.segment_trials` writes) and loads each atomic, variable-length
``.npz`` time series (keys per :data:`wallflower.contract.NPZ_KEYS`). It:

  * filters by modality / style / perspective / participant,
  * appends the per-step time-delta ``dt`` as the *final* input feature so that
    ``input_dim == contract.expected_feature_dim(modality, with_dt=True)``
    (paper: dt is consumed as an extra temporal feature),
  * builds a deterministic participant ``label <-> index`` vocabulary,
  * applies a :class:`Standardiser` fitted on the *train split only* (the paper
    standardises features and does no other handcrafted preprocessing),
  * provides :func:`collate_fn` that pads sequences and returns lengths for
    ``pack_padded_sequence``,
  * provides :func:`stratified_split` for a reproducible, seeded 80/20 split.

The Standardiser is imported from :mod:`parsers.normalise` (its canonical home);
a byte-for-byte compatible fallback is defined here so the model stack is
runnable end-to-end before the parsers module lands.
"""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wallflower import contract

# --------------------------------------------------------------------------- #
# Standardiser: prefer the canonical parsers.normalise one, else fall back.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised once parsers.normalise exists
    from parsers.normalise import Standardiser  # type: ignore
except Exception:  # ModuleNotFoundError or partial parsers package

    class Standardiser:
        """Zero-mean / unit-variance feature standardiser (train-fit only).

        Streaming fit so we never hold all sequences in memory at once. The
        appended dt column is standardised together with the signal features
        (it is just another input dimension to the LSTM). A small epsilon
        guards constant features (e.g. dt[0]==0 padding artefacts).
        """

        def __init__(self, eps: float = 1e-8) -> None:
            self.eps = float(eps)
            self.mean_: np.ndarray | None = None
            self.std_: np.ndarray | None = None
            self.n_features_: int | None = None

        def fit(self, sequences: Iterable[np.ndarray]) -> "Standardiser":
            count = 0
            ssum: np.ndarray | None = None
            ssq: np.ndarray | None = None
            for seq in sequences:
                arr = np.asarray(seq, dtype=np.float64)
                if arr.ndim != 2:
                    raise ValueError("each sequence must be [timesteps, features]")
                if ssum is None:
                    ssum = arr.sum(axis=0)
                    ssq = (arr * arr).sum(axis=0)
                    self.n_features_ = arr.shape[1]
                else:
                    if arr.shape[1] != self.n_features_:
                        raise ValueError("inconsistent feature width across samples")
                    ssum += arr.sum(axis=0)
                    ssq += (arr * arr).sum(axis=0)
                count += arr.shape[0]
            if count == 0 or ssum is None:
                raise ValueError("cannot fit Standardiser on empty data")
            mean = ssum / count
            var = np.maximum(ssq / count - mean * mean, 0.0)
            self.mean_ = mean.astype(np.float32)
            self.std_ = (np.sqrt(var) + self.eps).astype(np.float32)
            return self

        def transform(self, seq: np.ndarray) -> np.ndarray:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Standardiser must be fit before transform")
            arr = np.asarray(seq, dtype=np.float32)
            return (arr - self.mean_) / self.std_

        def to_dict(self) -> dict:
            return {
                "eps": self.eps,
                "mean": None if self.mean_ is None else self.mean_.tolist(),
                "std": None if self.std_ is None else self.std_.tolist(),
                "n_features": self.n_features_,
            }

        @classmethod
        def from_dict(cls, d: dict) -> "Standardiser":
            obj = cls(eps=d.get("eps", 1e-8))
            obj.mean_ = None if d.get("mean") is None else np.asarray(d["mean"], np.float32)
            obj.std_ = None if d.get("std") is None else np.asarray(d["std"], np.float32)
            obj.n_features_ = d.get("n_features")
            return obj


__all__ = [
    "Standardiser",
    "SampleRow",
    "BFIdDataset",
    "collate_fn",
    "stratified_split",
    "load_index",
]


# --------------------------------------------------------------------------- #
# Index loading
# --------------------------------------------------------------------------- #
@dataclass
class SampleRow:
    """One row of the processed sample index (one atomic .npz)."""

    npz_path: Path
    participant: str
    style: str
    perspective: int
    modality: str


def load_index(
    data_root: Path | str,
    *,
    modality: str | None = None,
    styles: Sequence[str] | None = None,
    perspectives: Sequence[int] | None = None,
    participants: Sequence[str] | None = None,
) -> list[SampleRow]:
    """Read ``samples.parquet`` and return filtered :class:`SampleRow` list.

    Falls back to globbing ``data/processed/samples/*.npz`` (decoding the
    canonical ``sample_basename``) when the parquet index is absent, so the
    model stack runs against parser output even before the index is built.
    """
    data_root = Path(data_root)
    parquet = data_root / "processed" / contract.SAMPLES_PARQUET
    rows: list[SampleRow] = []

    if parquet.exists():
        import pandas as pd  # local import: heavy, [ml] extra only

        df = pd.read_parquet(parquet)
        for rec in df.to_dict(orient="records"):
            # The canonical index column for the participant id is contract's
            # `label` (== participant); accept `participant` as a legacy alias.
            pid = rec.get("participant") or rec.get("label")
            npz_path = rec.get("npz_path") or rec.get("path")
            if npz_path is None:
                # reconstruct from canonical naming
                npz_path = contract.processed_sample_path(
                    data_root, pid, rec["style"],
                    str(rec["trial"]).zfill(3), rec["modality"], int(rec["perspective"]),
                )
            p = Path(npz_path)
            if not p.is_absolute():
                p = data_root / "processed" / "samples" / p.name
            rows.append(SampleRow(
                npz_path=p,
                participant=str(pid),
                style=str(rec["style"]),
                perspective=int(rec["perspective"]),
                modality=str(rec["modality"]),
            ))
    else:
        rows = _scan_npz_dir(data_root)

    def keep(r: SampleRow) -> bool:
        if modality is not None and r.modality != modality:
            return False
        if styles is not None and r.style not in set(styles):
            return False
        if perspectives is not None and r.perspective not in set(perspectives):
            return False
        if participants is not None and r.participant not in set(participants):
            return False
        # Skip index rows whose .npz is gone (stale index / partial capture):
        # never crash training on a missing sample file — just drop it.
        if not r.npz_path.exists():
            return False
        return True

    return [r for r in rows if keep(r)]


def _scan_npz_dir(data_root: Path) -> list[SampleRow]:
    """Recover the index by decoding sample_basename from npz filenames."""
    samples_dir = Path(data_root) / "processed" / "samples"
    out: list[SampleRow] = []
    if not samples_dir.exists():
        return out
    for npz in sorted(samples_dir.glob("*.npz")):
        # P001_normal_001_bfi_p1  ->  participant, style, trial, modality, p#
        stem = npz.stem
        parts = stem.split("_")
        if len(parts) < 5:
            continue
        participant, style, _trial, modality, ptag = parts[0], parts[1], parts[2], parts[3], parts[4]
        if not ptag.startswith("p"):
            continue
        try:
            perspective = int(ptag[1:])
        except ValueError:
            continue
        out.append(SampleRow(npz, participant, style, perspective, modality))
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BFIdDataset(Dataset):
    """Variable-length sample dataset for the baseline LSTM.

    Each item is ``(x, length, label_idx, meta)`` where ``x`` is a float32
    ``[T, input_dim]`` tensor (features + appended dt), ``length`` is ``T``, and
    ``meta`` carries participant/style/perspective for grouped evaluation.

    The label vocabulary is built from the participant ids present in ``rows``
    (sorted for determinism). Pass ``label_to_idx`` to reuse a fixed vocab
    (e.g. so the test set matches the trained model's classes). Pass a fitted
    ``standardiser`` to apply train-fit statistics to a held-out split.
    """

    def __init__(
        self,
        rows: Sequence[SampleRow],
        *,
        modality: str | None = None,
        label_to_idx: dict[str, int] | None = None,
        standardiser: Standardiser | None = None,
        append_dt: bool = True,
    ) -> None:
        if not rows:
            raise ValueError("BFIdDataset received no samples (check filters)")
        self.rows = list(rows)
        self.append_dt = append_dt

        # Modality consistency (input_dim depends on it).
        modalities = {r.modality for r in self.rows}
        if modality is not None:
            modalities = {modality}
            self.rows = [r for r in self.rows if r.modality == modality]
            if not self.rows:
                raise ValueError(f"no samples for modality={modality!r}")
        if len(modalities) != 1:
            raise ValueError(
                f"dataset mixes modalities {modalities}; filter to one "
                "(input_dim differs per modality)"
            )
        self.modality = next(iter(modalities))
        self.input_dim = contract.expected_feature_dim(
            self.modality, with_dt=self.append_dt
        )

        # Label vocab.
        if label_to_idx is None:
            labels = sorted({r.participant for r in self.rows})
            label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        self.label_to_idx = dict(label_to_idx)
        self.idx_to_label = {i: lbl for lbl, i in self.label_to_idx.items()}

        self.standardiser = standardiser

    # -- core sequence loader --------------------------------------------- #
    def _load_xdt(self, row: SampleRow) -> tuple[np.ndarray, np.ndarray]:
        """Load one sample's feature matrix [T, features] and dt vector [T].

        Features and dt are kept SEPARATE so the (train-fit) Standardiser can
        standardise the modality features while applying its own policy to dt
        (per parsers.normalise); they are concatenated into the model input only
        in :meth:`__getitem__`.
        """
        with np.load(row.npz_path, allow_pickle=True) as npz:
            x = np.asarray(npz["x"], dtype=np.float32)
            if x.ndim != 2:
                raise ValueError(f"{row.npz_path}: x must be 2-D [T, features]")
            expected_feat = contract.FEATURE_DIMS[self.modality]
            if x.shape[1] != expected_feat:
                raise ValueError(
                    f"{row.npz_path}: x has {x.shape[1]} features, "
                    f"expected {expected_feat} for {self.modality}"
                )
            dt = np.asarray(npz["dt"], dtype=np.float32).reshape(-1)
            if dt.shape[0] != x.shape[0]:
                raise ValueError(
                    f"{row.npz_path}: dt length {dt.shape[0]} != T {x.shape[0]}"
                )
        return x, dt

    def raw_sequences(self) -> Iterable[np.ndarray]:
        """Yield un-standardised feature matrices [T, features] (for fitting stats)."""
        for row in self.rows:
            yield self._load_xdt(row)[0]

    def fit_standardiser(self, dt_policy: str = "scale") -> Standardiser:
        """Fit a Standardiser on THIS dataset's samples (call on the train split).

        Standardises the modality features (zero-mean/unit-var); the appended dt
        column is handled by the canonical Standardiser's dt policy (default
        ``scale``: divide by the nominal sampling period — a constant, so no
        train/test leakage). Paper: feature standardisation only, no other
        handcrafted preprocessing.
        """
        std = Standardiser(modality=self.modality, dt_policy=dt_policy)
        if dt_policy == "standard":
            std.fit(self.raw_sequences(),
                    (self._load_xdt(r)[1] for r in self.rows))
        else:
            std.fit(self.raw_sequences())
        self.standardiser = std
        return std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        x, dt = self._load_xdt(row)
        if self.standardiser is not None:
            x = self.standardiser.transform(x)
            dt = self.standardiser.scale_dt(dt)
        if self.append_dt:
            x = np.concatenate([x, np.asarray(dt, dtype=np.float32)[:, None]], axis=1)
        x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        length = x_t.shape[0]
        label_idx = self.label_to_idx[row.participant]
        meta = {
            "participant": row.participant,
            "style": row.style,
            "perspective": row.perspective,
            "modality": row.modality,
        }
        return x_t, length, label_idx, meta


# --------------------------------------------------------------------------- #
# Collation (pad + lengths for pack_padded_sequence)
# --------------------------------------------------------------------------- #
def collate_fn(batch: list):
    """Pad a list of ``(x, length, label, meta)`` into batched tensors.

    Returns ``(padded [B, T_max, F], lengths [B], labels [B], metas list)``.
    Sequences are *not* pre-sorted: :class:`BFIdLSTM` packs with
    ``enforce_sorted=False``.
    """
    xs, lengths, labels, metas = zip(*batch, strict=True)
    b = len(xs)
    feat = xs[0].shape[1]
    t_max = max(int(length) for length in lengths)
    padded = torch.zeros((b, t_max, feat), dtype=torch.float32)
    for i, x in enumerate(xs):
        t = x.shape[0]
        padded[i, :t] = x
    lengths_t = torch.tensor([int(length) for length in lengths], dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded, lengths_t, labels_t, list(metas)


# --------------------------------------------------------------------------- #
# Reproducible stratified split
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    train: list[SampleRow] = field(default_factory=list)
    test: list[SampleRow] = field(default_factory=list)


def stratified_split(
    rows: Sequence[SampleRow],
    *,
    test_frac: float = 0.2,
    seed: int = 1337,
) -> SplitResult:
    """Seeded stratified split (default 80/20) keyed on participant.

    Stratifies by participant so every identity appears in both splits when it
    has >= 2 samples (the paper trains/tests per-identity). Deterministic for a
    given ``seed``. A participant with a single sample is placed in train.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    by_label: dict[str, list[SampleRow]] = defaultdict(list)
    for r in rows:
        by_label[r.participant].append(r)

    rng = random.Random(seed)
    res = SplitResult()
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n = len(group)
        if n == 1:
            res.train.extend(group)
            continue
        n_test = max(1, round(n * test_frac))
        n_test = min(n_test, n - 1)  # always leave >=1 for train
        res.test.extend(group[:n_test])
        res.train.extend(group[n_test:])
    # stable order for reproducibility
    res.train.sort(key=lambda r: r.npz_path.name)
    res.test.sort(key=lambda r: r.npz_path.name)
    return res
