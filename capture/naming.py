"""capture.naming — single import surface for file/dir naming in capture code.

This is a thin re-export layer over :mod:`wallflower.contract` so capture wrappers
have *one* place to import naming helpers from and can never drift from the
canonical layout. Do not redefine any constant here; only re-export and add the
small derived ``logs_dir`` helper that the capture wrappers need.

Canonical raw layout produced per trial (see contract)::

    data/raw/participant=P001/style=normal/trial=001/
        metadata.json
        csi_p1.raw csi_p2.raw csi_p3.raw csi_p4.raw
        bfi_recorder.pcapng
        logs/
"""

from __future__ import annotations

from pathlib import Path

# Re-export the canonical naming helpers unchanged.
from wallflower.contract import (
    bfi_recorder_name,
    csi_raw_name,
    metadata_name,
    raw_trial_dir,
)

__all__ = [
    "raw_trial_dir",
    "csi_raw_name",
    "bfi_recorder_name",
    "metadata_name",
    "logs_dir",
]

# Subdirectory name used for per-trial capture logs (pidfiles, stderr, notes).
LOGS_DIRNAME = "logs"


def logs_dir(trial_dir: Path | str) -> Path:
    """Return ``<trial_dir>/logs`` (not created here; caller mkdirs).

    The capture wrappers drop their pidfiles, tool stderr and any
    ``.MISSING_*`` marker notes into this directory so that the primary raw
    artifacts in ``trial_dir`` stay clean for the parsers.
    """
    return Path(trial_dir) / LOGS_DIRNAME
