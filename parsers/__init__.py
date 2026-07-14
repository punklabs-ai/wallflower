"""parsers/ — raw capture -> ML-ready variable-length npz time series.

Realises the ``data/raw -> data/processed`` transformation of the BFId lab
(Todt, Morsbach, Strufe, CCS '25). Each *atomic sample* is one
(participant, style, trial, modality, perspective) variable-length time series
written to ``data/processed/samples/<sample_basename>.npz`` with the canonical
``contract.NPZ_KEYS`` schema, plus a per-sample row appended to
``data/processed/samples.parquet`` (the index models/dataset.py reads).

Modules
-------
* :mod:`parsers.parse_bfi`   pcap/pcapng -> [T, 740] BFI angle series.
* :mod:`parsers.parse_csi`   PicoScenes .raw -> [T, 212] CSI phase/mag series.
* :mod:`parsers.segment_trials`  walk data/raw, drive the parsers, build index.
* :mod:`parsers.normalise`   train-only Standardiser (zero-mean/unit-var).

All modules import dims/paths/labels from :mod:`wallflower.contract` so naming never
drifts. numpy is required (``[ml]`` extra); real decoding uses scapy (BFI) and
the FeitCSI / PicoScenes backends (CSI) when available.
"""
from __future__ import annotations

__all__ = [
    "parse_bfi",
    "parse_csi",
    "segment_trials",
    "normalise",
]
