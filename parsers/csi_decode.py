"""parsers.csi_decode — self-contained AX210 CSI binary decoder (numpy only).

Reads the **FeitCSI** binary CSI capture format produced by the open-source
FeitCSI tool (https://github.com/KuskoSoft/FeitCSI), the recommended real
backend for AX210 CSI on this lab (the only AX210 CSI tool with a plausible
kernel-7.0 path; PicoScenes ships per-kernel prebuilt driver .debs and none
exists for kernel 7.0). It selects 53 subcarriers x 2 antennas, computes the
212 contract features (magnitude + phase), and builds ``dt`` from per-packet
timestamps. A deterministic FeitCSI-format encoder (seeded by sample name)
writes real FeitCSI bytes so the decoder is provable bit-for-bit with no
hardware (the inverse, mirroring how ``bfi_decode.encode_report`` proves
``decode_report``).

This module is **pure numpy** with no driver / toolbox dependency: the FeitCSI
format is a plain binary parse, so the whole real decode path is unit-testable
now. It performs NO I/O against radios and NEVER touches a driver.

FeitCSI binary format (the format this decoder targets)
-------------------------------------------------------
A capture file is a stream of records, each = a 272-byte little-endian header
followed by a raw CSI payload. Per-record header fields used here
(byte offsets, little-endian):

  * ``[0:4]``    u32  ``csi_data_size`` — payload size in bytes.
  * ``[8:12]``   u32  ``ftm_clock``     — FTM clock (informational).
  * ``[12:20]``  u64  ``timestamp``     — host clock in MICROSECONDS -> dt source.
  * ``[46]``     u8   ``num_rx``        — number of RX antennas.
  * ``[47]``     u8   ``num_tx``        — number of TX spatial streams.
  * ``[52:56]``  u32  ``num_subcarriers``.
  * ``[60:64]``  u32  ``rssi_tx1`` (informational).
  * ``[64:68]``  u32  ``rssi_tx2`` (informational).
  * ``[68:74]``  6B   ``src_mac`` (informational).
  * ``[92:96]``  u32  ``rate_flags`` (informational).

Payload: ``csi_data_size == 4 * num_rx * num_tx * num_subcarriers`` bytes, each
CSI value = 4 bytes = ``int16`` real (LE, bytes 0-1) + ``int16`` imag (LE,
bytes 2-3). Dimension ordering is subcarrier-outer: for each subcarrier, all
(rx, tx) pairs in row-major (rx outer, tx inner) order, then the next
subcarrier. So the natural reshape is ``[num_subcarriers, num_rx, num_tx]``.

212-feature computation (mirrors the contract + bfi_decode flatten convention)
-----------------------------------------------------------------------------
Per packet, select ``H53x2 = [53 subcarriers, 2 antennas]`` complex, then build
a ``[53, 2(ant), 2(comp)]`` array (comp 0 = magnitude, comp 1 = phase) and
flatten in C-order (subcarrier-major, then antenna, then component):

    feature[sc*4 + ant*2 + comp]     comp 0 = magnitude, 1 = phase

This mirrors ``bfi_decode.flatten_to_features`` (subcarrier-major C-order) and
yields exactly ``53 * 2 * 2 = 212`` features.

Phase handling: RAW magnitude (``np.abs``) + RAW phase (``np.angle``), matching
the BFId paper's deliberately simple "standardise-only" pipeline (no CSI
sanitisation / STO-SFO removal). An OPTIONAL ``unwrap`` flag (default off) is
provided so the choice is documented and reversible; do NOT enable CSI-ratio /
conjugate sanitisation here.

References: FeitCSI binary record format (KuskoSoft FeitCSI Python/Matlab
parsers); BFId paper (Todt/Morsbach/Strufe, CCS '25 sec. 5.1).
"""
from __future__ import annotations

import hashlib
import logging
import struct
from pathlib import Path

import numpy as np

from wallflower import contract

log = logging.getLogger("parsers.csi_decode")

PI = np.pi

# Contract feature geometry (import; never redefine).
NSUB = contract.CSI_SUBCARRIERS   # 53
NANT = contract.CSI_ANTENNAS      # 2
NCOMP = contract.CSI_COMPONENTS   # 2 (magnitude, phase)
NFEAT = contract.CSI_FEATURES     # 212 = 53 * 2 * 2

# --------------------------------------------------------------------------- #
# FeitCSI binary record layout (little-endian byte offsets within the header)
# Single named table: change ONLY this if a real capture disagrees.
# --------------------------------------------------------------------------- #
FEITCSI_HEADER_LEN = 272
_OFF_CSI_DATA_SIZE = 0      # u32
_OFF_FTM_CLOCK = 8          # u32
_OFF_TIMESTAMP = 12         # u64, microseconds
_OFF_NUM_RX = 46            # u8
_OFF_NUM_TX = 47            # u8
_OFF_NUM_SUBCARRIERS = 52   # u32

# Comp index convention in the flattened feature vector.
COMP_MAG = 0
COMP_PHASE = 1


# --------------------------------------------------------------------------- #
# Antenna / subcarrier selection (single named helpers; lab decisions live here)
# --------------------------------------------------------------------------- #
def select_antennas(H: np.ndarray, info: dict | None = None) -> np.ndarray:
    """Select the 2 contract "antennas" from ``H[num_sub, num_rx, num_tx]``.

    The AP-CSI link on ch37 is SU; we treat the **2 RX antennas of the AX210**
    as the paper's "2 antennas" and fix ``tx = 0``: ``H[:, 0:2, 0]`` ->
    ``[num_sub, 2]``. If a capture yields only 1 RX, RX index 1 is duplicated
    from RX 0 and ``rx_duplicated`` is flagged (mirrors the bfi zero-pad audit).
    Change ONLY this helper to alter the lab's rx/tx pick.
    """
    if H.ndim != 3:
        raise ValueError(f"expected [num_sub, num_rx, num_tx], got {H.shape}")
    num_sub, num_rx, num_tx = H.shape
    if num_rx >= NANT:
        H2 = H[:, 0:NANT, 0]
    elif num_rx == 1:
        H2 = np.repeat(H[:, 0:1, 0], NANT, axis=1)
        if info is not None:
            info["rx_duplicated"] = True
    else:
        raise ValueError(f"capture has num_rx={num_rx} < 1")
    return np.ascontiguousarray(H2)


def select_subcarriers(H2: np.ndarray, info: dict | None = None) -> np.ndarray:
    """Map ``H2[num_sub, 2]`` -> ``[53, 2]`` deterministically (audited).

    ``num_sub == 53`` -> pass through (preferred; a 20 MHz ch37 capture yields
    exactly 53 data subcarriers). ``num_sub > 53`` (e.g. 160 MHz gives 1992/2048
    tones) -> uniform ``np.linspace`` decimation (mirrors
    ``bfi_decode.map_to_contract_channels``), recording ``ns_raw``.
    ``num_sub < 53`` -> zero-pad the tail and flag. Always records the chosen
    mapping in ``info`` when ``num_sub != 53``.
    """
    if info is None:
        info = {}
    num_sub = H2.shape[0]
    if num_sub == NSUB:
        info.setdefault("subcarrier_map", "passthrough")
        return H2
    if num_sub > NSUB:
        idx = np.linspace(0, num_sub - 1, NSUB).round().astype(int)
        info["subcarrier_map"] = "decimate_linspace"
        info["ns_raw"] = int(num_sub)
        return np.ascontiguousarray(H2[idx])
    out = np.zeros((NSUB, H2.shape[1]), dtype=H2.dtype)
    out[:num_sub] = H2
    info["subcarrier_map"] = "zeropad"
    info["short_subcarriers"] = True
    info["ns_raw"] = int(num_sub)
    return out


def features_from_csi(H53x2: np.ndarray, *, unwrap: bool = False) -> np.ndarray:
    """``[53, 2]`` complex (one packet) -> length-212 float32 features.

    Builds ``[53, 2(ant), 2(comp)]`` with comp 0 = magnitude (``np.abs``),
    comp 1 = phase (``np.angle``, raw, in (-pi, pi]), then C-order flatten to
    ``feature[sc*4 + ant*2 + comp]``. ``unwrap`` (default off, paper-faithful)
    optionally applies ``np.unwrap`` along the subcarrier axis to the phase
    block only; off matches the paper's raw pipeline.
    """
    if H53x2.shape != (NSUB, NANT):
        raise ValueError(f"expected [{NSUB},{NANT}], got {H53x2.shape}")
    mag = np.abs(H53x2).astype(np.float32)           # [53, 2]
    phase = np.angle(H53x2).astype(np.float32)        # [53, 2] in (-pi, pi]
    if unwrap:
        phase = np.unwrap(phase, axis=0).astype(np.float32)
    stacked = np.empty((NSUB, NANT, NCOMP), dtype=np.float32)
    stacked[:, :, COMP_MAG] = mag
    stacked[:, :, COMP_PHASE] = phase
    return np.ascontiguousarray(stacked).reshape(NFEAT)


# --------------------------------------------------------------------------- #
# FeitCSI binary reader
# --------------------------------------------------------------------------- #
def _parse_feitcsi_header(hdr: bytes) -> dict | None:
    """Parse one 272-byte FeitCSI header -> field dict, or None if malformed."""
    if len(hdr) < FEITCSI_HEADER_LEN:
        return None
    csi_data_size = struct.unpack_from("<I", hdr, _OFF_CSI_DATA_SIZE)[0]
    timestamp_us = struct.unpack_from("<Q", hdr, _OFF_TIMESTAMP)[0]
    num_rx = hdr[_OFF_NUM_RX]
    num_tx = hdr[_OFF_NUM_TX]
    num_sub = struct.unpack_from("<I", hdr, _OFF_NUM_SUBCARRIERS)[0]
    return {
        "csi_data_size": int(csi_data_size),
        "timestamp_us": int(timestamp_us),
        "num_rx": int(num_rx),
        "num_tx": int(num_tx),
        "num_subcarriers": int(num_sub),
    }


def _header_consistent(h: dict) -> bool:
    """True if csi_data_size matches 4 * num_rx * num_tx * num_subcarriers and
    the counts are sane (used both as a sanity gate and a FeitCSI magic test)."""
    if h is None:
        return False
    nrx, ntx, nsub = h["num_rx"], h["num_tx"], h["num_subcarriers"]
    if nrx < 1 or ntx < 1 or nsub < 1:
        return False
    if nrx > 8 or ntx > 8 or nsub > 4096:
        return False
    return h["csi_data_size"] == 4 * nrx * ntx * nsub


def _csi_block_to_complex(payload: bytes, h: dict) -> np.ndarray:
    """Raw CSI payload -> ``[num_sub, num_rx, num_tx]`` complex64.

    Each value is int16 real + int16 imag (LE); subcarrier-outer ordering, with
    (rx, tx) pairs row-major (rx outer, tx inner) within a subcarrier.
    """
    nrx, ntx, nsub = h["num_rx"], h["num_tx"], h["num_subcarriers"]
    ints = np.frombuffer(payload, dtype="<i2", count=2 * nrx * ntx * nsub)
    real = ints[0::2].astype(np.float32)
    imag = ints[1::2].astype(np.float32)
    H = (real + 1j * imag).astype(np.complex64)
    return H.reshape(nsub, nrx, ntx)


def read_feitcsi_blocks(raw_path: Path) -> list[tuple[dict, np.ndarray]] | None:
    """Read a FeitCSI ``.raw``/``.dat`` -> list of ``(header, H[nsub,nrx,ntx])``.

    Returns ``None`` if the file does not look like FeitCSI (first header fails
    the consistency/magic check) so callers can fall through to other backends.
    Trailing truncated records are dropped (variable length is fine).
    """
    raw_path = Path(raw_path)
    data = raw_path.read_bytes()
    if len(data) < FEITCSI_HEADER_LEN:
        return None
    first = _parse_feitcsi_header(data[:FEITCSI_HEADER_LEN])
    if not _header_consistent(first):
        return None

    blocks: list[tuple[dict, np.ndarray]] = []
    pos = 0
    n = len(data)
    while pos + FEITCSI_HEADER_LEN <= n:
        h = _parse_feitcsi_header(data[pos:pos + FEITCSI_HEADER_LEN])
        if not _header_consistent(h):
            log.warning("FeitCSI record at offset %d inconsistent; stopping", pos)
            break
        pos += FEITCSI_HEADER_LEN
        size = h["csi_data_size"]
        if pos + size > n:
            log.warning("truncated FeitCSI payload at offset %d; dropping", pos)
            break
        payload = data[pos:pos + size]
        pos += size
        blocks.append((h, _csi_block_to_complex(payload, h)))
    if not blocks:
        return None
    return blocks


def decode_feitcsi(raw_path: Path, *, unwrap: bool = False,
                   info: dict | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Decode a FeitCSI capture -> ``(x[T,212], dt[T])`` or ``None``.

    Iterates records, selects 2 antennas + 53 subcarriers per packet, computes
    the 212 features, sorts packets chronologically by header timestamp and
    builds ``dt`` (microseconds -> seconds) with ``dt[0] == 0.0``. Surfaces the
    median packet rate in ``info`` for a sanity check against
    ``contract.NOMINAL_RATE_HZ['csi']``. Returns ``None`` if not FeitCSI.
    """
    if info is None:
        info = {}
    blocks = read_feitcsi_blocks(Path(raw_path))
    if blocks is None:
        return None

    rows: list[np.ndarray] = []
    times: list[float] = []
    for h, H in blocks:
        H2 = select_antennas(H, info=info)
        H53 = select_subcarriers(H2, info=info)
        rows.append(features_from_csi(H53, unwrap=unwrap))
        times.append(h["timestamp_us"] / 1e6)

    order = np.argsort(times)
    x = np.asarray(rows, dtype=np.float32)[order]
    ts = np.asarray(times, dtype=np.float64)[order]
    dt = np.diff(ts, prepend=ts[0]).astype(np.float32)
    dt[0] = 0.0

    if x.shape[0] >= 2:
        med = float(np.median(dt[1:]))
        info["median_rate_hz"] = (1.0 / med) if med > 0 else None
        info["nominal_rate_hz"] = contract.NOMINAL_RATE_HZ["csi"]
    info["n_packets"] = int(x.shape[0])
    return x, dt


# --------------------------------------------------------------------------- #
# FeitCSI-format encoder (inverse of the decoder; for bit-exact round-trip tests)
# --------------------------------------------------------------------------- #
def _seed_from_name(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def encode_feitcsi_block(H: np.ndarray, timestamp_us: int) -> bytes:
    """Encode one ``[num_sub, num_rx, num_tx]`` complex block -> FeitCSI record.

    Exact inverse of the reader: writes a 272-byte header (with consistent
    ``csi_data_size``, counts and ``timestamp``) then int16 real+imag pairs in
    subcarrier-outer, (rx, tx) row-major order. Values are rounded to int16.
    """
    nsub, nrx, ntx = H.shape
    csi_data_size = 4 * nrx * ntx * nsub
    hdr = bytearray(FEITCSI_HEADER_LEN)
    struct.pack_into("<I", hdr, _OFF_CSI_DATA_SIZE, csi_data_size)
    struct.pack_into("<Q", hdr, _OFF_TIMESTAMP, int(timestamp_us))
    hdr[_OFF_NUM_RX] = nrx
    hdr[_OFF_NUM_TX] = ntx
    struct.pack_into("<I", hdr, _OFF_NUM_SUBCARRIERS, nsub)

    flat = H.reshape(-1)
    pairs = np.empty(flat.size * 2, dtype="<i2")
    pairs[0::2] = np.round(flat.real).astype("<i2")
    pairs[1::2] = np.round(flat.imag).astype("<i2")
    return bytes(hdr) + pairs.tobytes()


def make_feitcsi_blocks(sample_name: str, *, seconds: float = 6.0,
                        rate_hz: float | None = None, num_rx: int = 2,
                        num_tx: int = 1, num_sub: int = NSUB) -> bytes:
    """Deterministic FeitCSI capture bytes seeded by ``sample_name``.

    T is drawn from ~``rate_hz`` (default ~285 Hz) over ``seconds`` with jitter,
    so each sample has a different realistic variable length. Timestamps are
    strictly increasing (microseconds) -> ``dt >= 0`` with ``dt[0] == 0``. CSI
    values are bounded int16-range complex; NOT physically meaningful. The bytes
    are real FeitCSI records so :func:`decode_feitcsi` round-trips them.
    """
    rate = float(rate_hz if rate_hz is not None else contract.NOMINAL_RATE_HZ["csi"])
    rng = np.random.default_rng(_seed_from_name(sample_name))

    dur = seconds * (0.75 + 0.5 * rng.random())
    T = max(2, int(round(dur * rate)))

    base_us = (1.0 / rate if rate > 0 else 1.0 / 285.0) * 1e6
    gaps = base_us * (0.85 + 0.3 * rng.random(T))
    gaps[0] = 0.0
    ts_us = np.cumsum(gaps).round().astype(np.int64)

    out = bytearray()
    for i in range(T):
        real = rng.integers(-2000, 2000, size=(num_sub, num_rx, num_tx))
        imag = rng.integers(-2000, 2000, size=(num_sub, num_rx, num_tx))
        H = (real + 1j * imag).astype(np.complex64)
        out += encode_feitcsi_block(H, int(ts_us[i]))
    return bytes(out)
