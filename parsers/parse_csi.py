"""parsers.parse_csi — PicoScenes CSI ``.raw`` -> [T, 212] series.

Decodes a per-perspective ``csi_p<n>.raw`` capture into one variable-length CSI
time series (BFId paper ground-truth modality, Todt/Morsbach/Strufe CCS '25,
sec. 5.1):

    x  : float32 [T, 212]   (phase + magnitude) x 53 subcarriers x 2 antennas
    dt : float32 [T]        (seconds since previous frame; dt[0] == 0.0)

212 CSI features = ``contract.CSI_COMPONENTS`` (2: phase, magnitude) x
``contract.CSI_SUBCARRIERS`` (53) x ``contract.CSI_ANTENNAS`` (2).

Decode backends, in order
-------------------------
1. **FeitCSI binary capture** (primary backend) — a genuine FeitCSI
   ``.raw``/``.dat`` (272-byte header + int16-complex CSI). Decoded by
   :mod:`parsers.csi_decode` (pure numpy, no driver/toolbox), selecting 53
   subcarriers x 2 antennas and computing the 212 magnitude+phase features.
2. **PicoScenes python toolbox** (``picoscenes`` / ``PicoscenesToolbox``) if
   importable — secondary backend to read a genuine PicoScenes ``.raw`` and
   assemble phase/magnitude per subcarrier/antenna. Left as a clearly-marked
   stub (:func:`_decode_picoscenes`) returning ``None`` until wired up.

A capture that decodes with neither backend is an error — there is no
placeholder path. Offline parser of recorded captures.
numpy required.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wallflower import contract
from parsers import csi_decode

AGENT = "parse_csi"
MODALITY: contract.Modality = "csi"

# Optional PicoScenes python toolbox -----------------------------------------
try:  # pragma: no cover - depends on an optional, not-installed toolbox
    import picoscenes  # type: ignore  # noqa: F401

    _HAVE_PICOSCENES_TB = True
except Exception:  # noqa: BLE001
    _HAVE_PICOSCENES_TB = False


# --------------------------------------------------------------------------- #
# Decode: real FeitCSI binary capture (primary backend)
# --------------------------------------------------------------------------- #
def _decode_feitcsi(raw_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode a genuine FeitCSI binary ``.raw``/``.dat`` -> (x[T,212], dt[T]).

    Delegates to :func:`parsers.csi_decode.decode_feitcsi`, which parses the
    272-byte-header / int16-complex FeitCSI record stream, selects 53
    subcarriers x 2 antennas, computes the 212 magnitude+phase features in
    subcarrier-major C-order and derives dt from packet timestamps. This is a
    pure binary parse (NO driver/toolbox needed), so it is detectable and
    unit-testable without hardware. Returns ``None`` when the file is not in the
    FeitCSI format (first record fails the magic/consistency check), so the
    caller falls through to PicoScenes with NO schema change.
    """
    try:
        return csi_decode.decode_feitcsi(Path(raw_path))
    except Exception:  # noqa: BLE001 - any parse error -> fall through
        return None


# --------------------------------------------------------------------------- #
# Decode: real PicoScenes toolbox (secondary, import-gated stub)
# --------------------------------------------------------------------------- #
def _decode_picoscenes(raw_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode a genuine PicoScenes ``.raw`` via the python toolbox -> (x, dt).

    SECONDARY backend, import-gated on the (not-installed) ``picoscenes`` toolbox
    and left as a clearly-marked stub. A real implementation would read the
    PicoScenes frame log, take the CSI matrix per frame, then reuse
    :func:`csi_decode.select_antennas` / ``select_subcarriers`` /
    ``features_from_csi`` to build the 212-wide vector and derive dt from frame
    timestamps. Returns ``None`` until wired up so the caller falls back without
    a schema change.
    """
    if not _HAVE_PICOSCENES_TB:
        return None
    return None


def decode_csi(raw_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Best-effort decode of a real CSI ``.raw`` -> (x[T,212], dt[T]).

    Dispatch order: FeitCSI binary -> PicoScenes toolbox. Returns ``None`` if
    nothing could be decoded.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return None
    real = _decode_feitcsi(raw_path)
    if real is not None:
        return real
    real = _decode_picoscenes(raw_path)
    if real is not None:
        return real
    return None


# --------------------------------------------------------------------------- #
# npz writer (canonical schema)
# --------------------------------------------------------------------------- #
def _validate_sample(x: np.ndarray, dt: np.ndarray) -> None:
    feat = contract.FEATURE_DIMS[MODALITY]
    if x.ndim != 2 or x.shape[1] != feat:
        raise ValueError(f"CSI x must be [T, {feat}], got {x.shape}")
    if dt.shape != (x.shape[0],):
        raise ValueError(f"dt must be [T]=({x.shape[0]},), got {dt.shape}")
    if x.shape[0] > 0 and float(dt[0]) != 0.0:
        raise ValueError("dt[0] must be 0.0")


def write_npz(out_path: Path, x: np.ndarray, dt: np.ndarray, *,
              participant: str, style: str, perspective: int) -> Path:
    """Write one atomic CSI sample to ``out_path`` using ``contract.NPZ_KEYS``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.ascontiguousarray(x, dtype=np.float32)
    dt = np.ascontiguousarray(dt, dtype=np.float32)
    _validate_sample(x, dt)
    np.savez(
        out_path,
        x=x,
        dt=dt,
        label=np.asarray(participant),
        style=np.asarray(style),
        perspective=np.asarray(int(perspective), dtype=np.int64),
        modality=np.asarray(MODALITY),
    )
    written = out_path if out_path.suffix == ".npz" else out_path.with_suffix(".npz")
    if written != out_path and written.exists():
        written.replace(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def parse_csi(raw_path: Path | str, perspective: int, participant: str,
              style: str, trial: str, *, data_root: Path | str = "data") -> dict[str, Any]:
    """Parse one perspective's CSI ``.raw`` -> npz path + sample-info dict.

    Decodes real FeitCSI / PicoScenes data. Raises ``RuntimeError`` if the
    capture cannot be decoded — there is no placeholder fallback.
    """
    contract.validate_participant(participant)
    contract.validate_style(style)
    contract.validate_trial(trial)
    contract.validate_perspective(perspective)

    sample_name = contract.sample_basename(participant, style, trial, MODALITY, perspective)
    out_path = contract.processed_sample_path(
        data_root, participant, style, trial, MODALITY, perspective)

    decoded = decode_csi(Path(raw_path))
    if decoded is None:
        raise RuntimeError(
            f"no decodable CSI capture at {raw_path} (FeitCSI / PicoScenes)")
    x, dt = decoded

    write_npz(out_path, x, dt, participant=participant, style=style,
              perspective=perspective)

    info: dict[str, Any] = {
        "agent": AGENT,
        "sample": sample_name,
        "path": str(out_path),
        "label": participant,
        "style": style,
        "perspective": int(perspective),
        "modality": MODALITY,
        "n_timesteps": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "source_file": str(raw_path),
        "picoscenes_toolbox_available": _HAVE_PICOSCENES_TB,
    }
    return info


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"parsers.{AGENT}",
        description="Parse one perspective's CSI .raw into a [T,212] npz sample "
                    "(FeitCSI / PicoScenes; errors if undecodable).")
    p.add_argument("raw_path")
    p.add_argument("--participant", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--trial", required=True)
    p.add_argument("--perspective", type=int, required=True)
    p.add_argument("--data-root", default="data")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    info = parse_csi(args.raw_path, args.perspective, args.participant,
                     args.style, args.trial, data_root=args.data_root)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
