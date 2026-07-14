"""parsers.parse_bfi — compressed beamforming feedback (BFI) -> [T, 740] series.

Decodes the IEEE 802.11 compressed beamforming reports the BFId paper attacks
(Todt, Morsbach, Strufe, CCS '25, sec. 3 & 5.1) from the central
``bfi_recorder.pcapng`` into one variable-length time series **per perspective**:

    x  : float32 [T, 740]   (10 quantised beamforming angles x 74 channels)
    dt : float32 [T]        (seconds since previous report; dt[0] == 0.0)

740 BFI features = ``contract.BFI_ANGLES`` (10) x ``contract.BFI_CHANNELS`` (74).

Decode backend
--------------
Real decoding walks the pcapng for 802.11 management *action* frames carrying
VHT (category 21) / HE (category 30) compressed beamforming reports addressed
to/from the lab AP, dequantises the psi/phi angles and groups them per channel.
This uses **scapy** to read the frames when importable. scapy's stock dissector
does not expand the compressed-beamforming angle payload, so the byte-level
Givens-angle dequantiser lives in :mod:`parsers.bfi_decode`
(:func:`_decode_action_frame` delegates to it): it decodes VHT/HE reports to the
740-feature vector and returns ``None`` only on a mismatch or unsupported frame.

A pcapng that decodes to no BFI angle frames is an error — there is no
placeholder path.

This is an offline parser. It only reads files already recorded by
``capture/bfi_pcap.py`` (BSSID-pinned, no wildcard capture); it performs no
live sniffing.

numpy is required (``[ml]`` extra). scapy is optional.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wallflower import contract
from parsers import bfi_decode

AGENT = "parse_bfi"
MODALITY: contract.Modality = "bfi"

log = logging.getLogger(AGENT)

# Optional packet backend ----------------------------------------------------
try:  # pragma: no cover - depends on optional [capture] extra
    from scapy.all import PcapNgReader, PcapReader, Dot11  # type: ignore

    _HAVE_SCAPY = True
except Exception:  # noqa: BLE001 - any import failure means "no scapy"
    PcapReader = PcapNgReader = Dot11 = None  # type: ignore
    _HAVE_SCAPY = False


# --------------------------------------------------------------------------- #
# Real decode (scapy) — IEEE 802.11 VHT/HE compressed-beamforming dequantiser.
# Bit-level decode lives in :mod:`parsers.bfi_decode`; this layer handles frame
# iteration, perspective filtering and [T,740]+dt assembly.
# --------------------------------------------------------------------------- #
# 802.11 management action category numbers carrying compressed beamforming.
ACTION_CAT_VHT = bfi_decode.ACTION_CAT_VHT  # 21
ACTION_CAT_HE = bfi_decode.ACTION_CAT_HE    # 30


def _decode_action_frame(raw_bytes: bytes) -> np.ndarray | None:
    """Dequantise one VHT/HE compressed-beamforming action frame -> [740] vector.

    ``raw_bytes`` is the category-first action body (byte 0 = Category, byte 1 =
    Action code, byte 2: = MIMO Control + Compressed Beamforming Report). Decodes
    the Givens psi/phi angles (10 angles x 74 channels, subcarrier-major C-order:
    feature[sc*10 + a]) into a length-740 float32 vector, or ``None`` on any
    mismatch / truncation / non-target geometry so the frame is skipped. See
    :mod:`parsers.bfi_decode` (IEEE 802.11-2020 sec. 9.4.1.49/.50, 802.11ax HE
    MIMO Control + HE Compressed Beamforming).
    """
    return bfi_decode.decode_action_body(raw_bytes)


def _iter_pcap_frames(pcap_path: Path):
    """Yield ``(action_body_bytes, timestamp, ta_mac)`` for compressed-BF frames.

    Filters to management (type 0) Action / Action-No-Ack (subtype 13/14) frames
    whose action body carries category 21 (VHT) or 30 (HE). ``ta`` (Dot11.addr2)
    is the beamformee STA that sent the feedback (per-perspective key). scapy
    only expands ``Dot11Action`` for subtype 13 on read-back, so we read the
    category-first body directly from the Dot11 payload. Empty if no scapy.
    """
    if not _HAVE_SCAPY:
        return
    reader_cls = PcapNgReader if pcap_path.suffix == ".pcapng" else PcapReader
    skipped_htc = 0
    try:
        with reader_cls(str(pcap_path)) as rd:  # type: ignore[operator]
            for pkt in rd:
                try:
                    if not pkt.haslayer(Dot11):
                        continue
                    d = pkt[Dot11]
                    if d.type != 0 or d.subtype not in (13, 14):
                        continue
                    # +HTC guard: scapy mis-locates the body when FC Order is set.
                    if int(d.FCfield.order):
                        skipped_htc += 1
                        continue
                    body = bytes(d.payload)  # category-first action body
                    if len(body) < 2:
                        continue
                    if body[0] not in (ACTION_CAT_VHT, ACTION_CAT_HE):
                        continue
                    ts = float(pkt.time)
                    ta = d.addr2  # beamformee STA (perspective key)
                    yield body, ts, ta
                except Exception:  # noqa: BLE001 - skip undecodable frames
                    continue
    except Exception:  # noqa: BLE001 - unreadable/empty pcap -> no frames
        return
    if skipped_htc:
        log.warning("skipped %d +HTC (FC Order) frames in %s", skipped_htc, pcap_path)


def _perspective_mac(pcap_path: Path, perspective: int) -> str | None:
    """Beamformee STA MAC for ``perspective`` from the trial's metadata.json.

    Looks up a radio with role "bfi" matching the perspective. Returns the MAC
    (lower-case) or ``None`` when unavailable (then all frames are accepted, so
    the milestone path keeps working).
    """
    try:
        meta_path = Path(pcap_path).parent / contract.metadata_name()
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        for radio in meta.get("radios", []):
            if radio.get("role") == "bfi" and radio.get("perspective") == int(perspective):
                mac = radio.get("mac")
                return mac.lower() if mac else None
    except Exception:  # noqa: BLE001 - any metadata issue -> accept all
        return None
    return None


def decode_bfi(pcap_path: Path, perspective: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Attempt a real decode of one perspective's BFI series from ``pcap_path``.

    Returns ``(x[T,740], dt[T])`` on success or ``None`` if nothing could be
    decoded (no scapy, no frames matching the perspective MAC, or every frame
    returned ``None``). Frames are filtered to the beamformee STA MAC recorded in
    the trial metadata (all accepted if absent), decoded, then sorted
    chronologically; ``dt[0] == 0.0``.
    """
    if not _HAVE_SCAPY:
        return None
    target_mac = _perspective_mac(Path(pcap_path), perspective)
    rows: list[np.ndarray] = []
    times: list[float] = []
    for body, ts, ta in _iter_pcap_frames(Path(pcap_path)):
        if target_mac is not None and (ta or "").lower() != target_mac:
            continue
        vec = _decode_action_frame(body)
        if vec is None:
            continue
        rows.append(vec)
        times.append(ts)
    if not rows:
        return None
    order = np.argsort(times)
    x = np.asarray(rows, dtype=np.float32)[order]
    ts = np.asarray(times, dtype=np.float64)[order]
    dt = np.diff(ts, prepend=ts[0]).astype(np.float32)
    dt[0] = 0.0
    return x, dt


# --------------------------------------------------------------------------- #
# npz writer (canonical schema)
# --------------------------------------------------------------------------- #
def _validate_sample(x: np.ndarray, dt: np.ndarray) -> None:
    feat = contract.FEATURE_DIMS[MODALITY]
    if x.ndim != 2 or x.shape[1] != feat:
        raise ValueError(f"BFI x must be [T, {feat}], got {x.shape}")
    if dt.shape != (x.shape[0],):
        raise ValueError(f"dt must be [T]=({x.shape[0]},), got {dt.shape}")
    if x.shape[0] > 0 and float(dt[0]) != 0.0:
        raise ValueError("dt[0] must be 0.0")


def write_npz(out_path: Path, x: np.ndarray, dt: np.ndarray, *,
              participant: str, style: str, perspective: int) -> Path:
    """Write one atomic BFI sample to ``out_path`` using ``contract.NPZ_KEYS``."""
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
    # np.savez appends .npz; normalise back to the requested path name.
    written = out_path if out_path.suffix == ".npz" else out_path.with_suffix(".npz")
    if written != out_path and written.exists():
        written.replace(out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Public entrypoint
# --------------------------------------------------------------------------- #
def parse_bfi(pcap_path: Path | str, perspective: int, participant: str,
              style: str, trial: str, *, data_root: Path | str = "data") -> dict[str, Any]:
    """Parse one perspective's BFI series from ``pcap_path`` -> npz path + info.

    Runs a real scapy decode. Raises ``RuntimeError`` if no BFI angle frames can
    be decoded — there is no placeholder fallback. Returns a sample-info dict
    (consumed by ``segment_trials`` to build the parquet index).
    """
    contract.validate_participant(participant)
    contract.validate_style(style)
    contract.validate_trial(trial)
    contract.validate_perspective(perspective)

    sample_name = contract.sample_basename(participant, style, trial, MODALITY, perspective)
    out_path = contract.processed_sample_path(
        data_root, participant, style, trial, MODALITY, perspective)

    decoded = decode_bfi(Path(pcap_path), perspective)
    if decoded is None:
        raise RuntimeError(
            f"no decodable BFI frames in {pcap_path} "
            f"(perspective {perspective}; scapy available={_HAVE_SCAPY})")
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
        "source_file": str(pcap_path),
        "scapy_available": _HAVE_SCAPY,
    }
    return info


# --------------------------------------------------------------------------- #
# CLI (single-file convenience; segment_trials is the batch entrypoint)
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"parsers.{AGENT}",
        description="Parse one perspective's BFI series from a pcapng into a "
                    "[T,740] npz sample (errors if no BFI frames decode).")
    p.add_argument("pcap_path")
    p.add_argument("--participant", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--trial", required=True)
    p.add_argument("--perspective", type=int, required=True)
    p.add_argument("--data-root", default="data")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    info = parse_bfi(args.pcap_path, args.perspective, args.participant,
                     args.style, args.trial, data_root=args.data_root)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
