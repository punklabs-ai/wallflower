"""parsers.normalise — train-only standardisation utilities.

The BFId paper (Todt/Morsbach/Strufe, CCS '25, sec. 5.2) applies *standardisation
only* — zero-mean / unit-variance per feature — with no handcrafted preprocessing
(no filtering, PCA, hand-built statistics). This module provides a
:class:`Standardiser` that:

  * FITS its mean/std on the TRAINING split ONLY (fit on train, transform all),
    to avoid leaking test/val statistics — call :meth:`fit` (or
    :meth:`fit_from_npz`) with train samples, then :meth:`transform` everything;
  * saves / loads its fitted stats (npz preferred, json fallback) so
    ``models/dataset.py`` can reuse the exact same train-fitted transform at
    eval time;
  * standardises the ``contract.FEATURE_DIMS[modality]`` data features (740 BFI /
    212 CSI). The appended per-sample ``dt`` column is handled SEPARATELY and is
    NOT standardised with the data features by default — see below.

dt column handling (documented choice)
--------------------------------------
``dt`` is a temporal inter-arrival channel with very different units/scale from
the quantised angles / phase-magnitude features, and ``dt[0]`` is always 0.0.
Standardising it jointly with the 740/212 features would distort both. The
chosen policy, exposed via :class:`DtPolicy`:

  * ``"none"``    : leave dt untouched (raw seconds). DEFAULT.
  * ``"scale"``   : divide dt by a fixed nominal period (1 / NOMINAL_RATE_HZ) so
                    it is O(1) without using any data-derived statistics — still
                    train-leak-free since the scale is a constant, not fitted.
  * ``"standard"``: zero-mean/unit-var dt using TRAIN-fitted dt stats (kept
                    distinct from the feature stats). Opt-in only.

The Standardiser only ever fits/transforms the FEATURE matrix; dt scaling is a
separate explicit step (:meth:`scale_dt`) so the model/dataset layer decides how
to fold dt back in (it is appended as the final input column per the contract).

numpy required (``[ml]`` extra).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from wallflower import contract

DtPolicy = Literal["none", "scale", "standard"]
DT_POLICIES: tuple[DtPolicy, ...] = ("none", "scale", "standard")

# Numerical floor so a zero-variance feature does not blow up the transform.
_EPS = 1e-8


@dataclass
class Standardiser:
    """Per-feature zero-mean/unit-var standardiser, fit on TRAIN only.

    Attributes
    ----------
    modality   : "bfi" | "csi" — sets the expected feature width (validated).
    mean       : [features] float64 fitted feature means (None until fit).
    std        : [features] float64 fitted feature stds (None until fit).
    dt_policy  : how the dt column should be treated (see module docstring).
    dt_mean / dt_std : fitted dt stats, only when dt_policy == "standard".
    n_seen     : number of timesteps the stats were fit over (provenance).
    """

    modality: contract.Modality
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    dt_policy: DtPolicy = "none"
    dt_mean: float | None = None
    dt_std: float | None = None
    n_seen: int = 0

    # ------------------------------------------------------------------ #
    @property
    def n_features(self) -> int:
        return contract.FEATURE_DIMS[self.modality]

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def _check_width(self, x: np.ndarray) -> None:
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"{self.modality} features must be [T, {self.n_features}], "
                f"got {x.shape}")

    # ------------------------------------------------------------------ #
    # Fitting (TRAIN ONLY)
    # ------------------------------------------------------------------ #
    def fit(self, feature_matrices: Iterable[np.ndarray],
            dt_vectors: Iterable[np.ndarray] | None = None) -> "Standardiser":
        """Fit feature mean/std over the concatenation of TRAIN samples.

        ``feature_matrices`` is an iterable of per-sample ``x`` arrays
        ([T_i, features]); they are accumulated with a streaming sum so very
        large train sets need not be concatenated in memory. Pass ``dt_vectors``
        only when ``dt_policy == 'standard'`` to fit dt stats too.
        """
        feat = self.n_features
        count = 0
        s1 = np.zeros(feat, dtype=np.float64)
        s2 = np.zeros(feat, dtype=np.float64)
        for x in feature_matrices:
            x = np.asarray(x, dtype=np.float64)
            self._check_width(x)
            if x.shape[0] == 0:
                continue
            s1 += x.sum(axis=0)
            s2 += np.square(x).sum(axis=0)
            count += x.shape[0]
        if count == 0:
            raise ValueError("cannot fit Standardiser on zero timesteps")
        mean = s1 / count
        var = np.maximum(s2 / count - np.square(mean), 0.0)
        self.mean = mean
        self.std = np.sqrt(var) + _EPS
        self.n_seen = count

        if self.dt_policy == "standard":
            if dt_vectors is None:
                raise ValueError(
                    "dt_policy='standard' requires dt_vectors to fit dt stats")
            dt_s1 = 0.0
            dt_s2 = 0.0
            dt_n = 0
            for d in dt_vectors:
                d = np.asarray(d, dtype=np.float64).ravel()
                dt_s1 += float(d.sum())
                dt_s2 += float(np.square(d).sum())
                dt_n += d.shape[0]
            if dt_n == 0:
                raise ValueError("cannot fit dt stats on zero timesteps")
            self.dt_mean = dt_s1 / dt_n
            self.dt_std = (dt_s2 / dt_n - (dt_s1 / dt_n) ** 2) ** 0.5 + _EPS
        return self

    @classmethod
    def fit_from_npz(cls, modality: contract.Modality, npz_paths: Iterable[Path | str],
                     *, dt_policy: DtPolicy = "none") -> "Standardiser":
        """Convenience: fit from a list of TRAIN-split sample npz paths.

        Reads each npz's ``x`` (and ``dt`` when ``dt_policy=='standard'``). Only
        samples whose ``modality`` matches are used.
        """
        std = cls(modality=modality, dt_policy=dt_policy)
        paths = [Path(p) for p in npz_paths]

        def _feat_iter():
            for p in paths:
                with np.load(p, allow_pickle=False) as z:
                    if str(z["modality"]) != modality:
                        continue
                    yield z["x"]

        def _dt_iter():
            for p in paths:
                with np.load(p, allow_pickle=False) as z:
                    if str(z["modality"]) != modality:
                        continue
                    yield z["dt"]

        std.fit(_feat_iter(), _dt_iter() if dt_policy == "standard" else None)
        return std

    # ------------------------------------------------------------------ #
    # Transforming
    # ------------------------------------------------------------------ #
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardise a feature matrix [T, features] -> float32 [T, features]."""
        if not self.fitted:
            raise RuntimeError("Standardiser not fitted; call fit() on TRAIN first")
        x = np.asarray(x, dtype=np.float64)
        self._check_width(x)
        out = (x - self.mean) / self.std
        return out.astype(np.float32)

    def scale_dt(self, dt: np.ndarray) -> np.ndarray:
        """Apply the configured dt policy to a dt vector -> float32 [T].

        ``none`` returns dt unchanged; ``scale`` divides by the nominal period
        (a constant, no train leakage); ``standard`` applies fitted dt stats.
        """
        dt = np.asarray(dt, dtype=np.float64).ravel()
        if self.dt_policy == "none":
            return dt.astype(np.float32)
        if self.dt_policy == "scale":
            period = 1.0 / contract.NOMINAL_RATE_HZ[self.modality]
            return (dt / period).astype(np.float32)
        # standard
        if self.dt_mean is None or self.dt_std is None:
            raise RuntimeError("dt stats not fitted; fit() with dt_vectors first")
        return ((dt - self.dt_mean) / self.dt_std).astype(np.float32)

    def transform_sample(self, x: np.ndarray, dt: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
        """Transform both the feature matrix and the dt column of one sample."""
        return self.transform(x), self.scale_dt(dt)

    # ------------------------------------------------------------------ #
    # Persistence (npz preferred, json fallback)
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str) -> Path:
        """Save fitted stats. ``.json`` -> json, anything else -> npz."""
        if not self.fitted:
            raise RuntimeError("refusing to save an unfitted Standardiser")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            payload = {
                "modality": self.modality,
                "n_features": self.n_features,
                "mean": self.mean.tolist(),  # type: ignore[union-attr]
                "std": self.std.tolist(),    # type: ignore[union-attr]
                "dt_policy": self.dt_policy,
                "dt_mean": self.dt_mean,
                "dt_std": self.dt_std,
                "n_seen": int(self.n_seen),
            }
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            np.savez(
                path,
                modality=np.asarray(self.modality),
                mean=self.mean,
                std=self.std,
                dt_policy=np.asarray(self.dt_policy),
                dt_mean=np.asarray(np.nan if self.dt_mean is None else self.dt_mean),
                dt_std=np.asarray(np.nan if self.dt_std is None else self.dt_std),
                n_seen=np.asarray(int(self.n_seen), dtype=np.int64),
            )
            if path.suffix != ".npz" and path.with_suffix(".npz").exists():
                path.with_suffix(".npz").replace(path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Standardiser":
        """Load fitted stats saved by :meth:`save`."""
        path = Path(path)
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                modality=payload["modality"],
                mean=np.asarray(payload["mean"], dtype=np.float64),
                std=np.asarray(payload["std"], dtype=np.float64),
                dt_policy=payload.get("dt_policy", "none"),
                dt_mean=payload.get("dt_mean"),
                dt_std=payload.get("dt_std"),
                n_seen=int(payload.get("n_seen", 0)),
            )
        with np.load(path, allow_pickle=False) as z:
            dt_mean = float(z["dt_mean"]) if "dt_mean" in z else None
            dt_std = float(z["dt_std"]) if "dt_std" in z else None
            return cls(
                modality=str(z["modality"]),  # type: ignore[arg-type]
                mean=np.asarray(z["mean"], dtype=np.float64),
                std=np.asarray(z["std"], dtype=np.float64),
                dt_policy=str(z["dt_policy"]) if "dt_policy" in z else "none",  # type: ignore[arg-type]
                dt_mean=None if dt_mean is None or np.isnan(dt_mean) else dt_mean,
                dt_std=None if dt_std is None or np.isnan(dt_std) else dt_std,
                n_seen=int(z["n_seen"]) if "n_seen" in z else 0,
            )
