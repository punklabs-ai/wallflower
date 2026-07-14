"""parsers.bfi_decode — self-contained IEEE 802.11 VHT/HE compressed
beamforming (CBF) report decoder + encoder (numpy only).

This module implements the bit-level decode of the compressed beamforming
report carried in 802.11ac (VHT, Action category 21) and 802.11ax (HE, Action
No Ack category 30) management action frames, dequantising the Givens-rotation
psi/phi angles into radians. The target case is HE, SU, 6 GHz, 160 MHz,
Nr=4, Nc=2 -> 10 angles/subcarrier (see ``wallflower.contract``).

It also provides the exact INVERSE (quantiser + encoder + frame builder) so the
decoder can be proven bit-exact with no hardware (round-trip tests).

References (IEEE Std 802.11-2020 / 802.11ax-2021):
  * VHT MIMO Control field             -- 802.11-2020 sec. 9.4.1.49
  * VHT Compressed Beamforming Report  -- 802.11-2020 sec. 9.4.1.50
  * VHT Action (category 21)           -- 802.11-2020 sec. 9.6.x
  * HE MIMO Control field              -- 802.11ax-2021 sec. 9.4.1.65
  * HE Compressed Beamforming/CQI      -- 802.11ax-2021 (HE Action No Ack, cat 30)

Bit order is LSB-first at two levels (the empirically-correct convention for
AX210 / Broadcom commodity captures, matching the Wi-BFI ``LSB=True`` setting):
each byte is reversed bit-wise before concatenation, and each sliced angle field
is reversed again before integer conversion (see ``_bitstring_lsb`` /
``_read_angle`` / ``_pack_angles``).

Pure functions only. No scapy, no I/O.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from wallflower import contract

log = logging.getLogger("parsers.bfi_decode")

PI = np.pi

# --------------------------------------------------------------------------- #
# Action frame constants (mirrored in parse_bfi)
# --------------------------------------------------------------------------- #
ACTION_CAT_VHT = 21          # 802.11ac, VHT Action
ACTION_CAT_HE = 30           # 802.11ax, HE Action (No Ack)
VHT_ACTION_COMPRESSED_BF = 0     # VHT Action code: VHT Compressed Beamforming
HE_ACTION_COMPRESSED_BF = 0      # HE Action code: HE Compressed Beamforming And CQI
                                 # (802.11ax-2021 Table 9-623). Confirmed against a
                                 # real AX210 capture (2026-06-02): action byte == 0.
HE_ACTION_CODES = (0,)           # spec defines only 0; observed 0 over the air.

# Contract feature geometry (import; never redefine).
NA = contract.BFI_ANGLES         # 10
NCH = contract.BFI_CHANNELS      # 74
NFEAT = contract.BFI_FEATURES    # 740

# Feedback type codes.
FB_SU = 0
FB_MU = 1
FB_CQI = 2

# MIMO Control widths (bytes).
MC_LEN = {"VHT": 3, "HE": 5}

# --------------------------------------------------------------------------- #
# MIMO Control bitfield maps -- single named tables: (lo, width).
# A one-line correction here fixes every call site (per spec sec. 2).
# --------------------------------------------------------------------------- #
# VHT MIMO Control -- 24 bits (802.11-2020 sec. 9.4.1.49).
VHT_MC_BITS: dict[str, tuple[int, int]] = {
    "nc_index": (0, 3),                 # Nc = val + 1
    "nr_index": (3, 3),                 # Nr = val + 1
    "bw": (6, 2),                       # 0=20,1=40,2=80,3=160/80+80
    "ng": (8, 2),                       # 0->Ng=1,1->Ng=2,2->Ng=4,3=reserved
    "codebook_info": (10, 1),
    "feedback_type": (11, 1),           # 0=SU,1=MU
    "remaining_segments": (12, 3),
    "first_segment": (15, 1),
    "sounding_token": (18, 6),
}
# HE MIMO Control -- 40 bits (802.11ax sec. 9.4.1.65). Medium-confidence byte map
# (see spec sec. 2); correct ONLY this table if a real capture disagrees.
HE_MC_BITS: dict[str, tuple[int, int]] = {
    "nc_index": (0, 3),                 # Nc = val + 1
    "nr_index": (3, 3),                 # Nr = val + 1
    "bw": (6, 2),                       # 0=20,1=40,2=80,3=160/80+80
    "ng": (8, 1),                       # 0->Ng=4,1->Ng=16 (HE allows {4,16})
    "codebook_info": (9, 1),
    "feedback_type": (10, 2),           # 0=SU,1=MU,2=CQI
    "remaining_segments": (12, 3),
    "first_segment": (15, 1),
    "ru_start": (16, 7),
    "ru_end": (23, 7),
    "sounding_token": (30, 6),
    "disambiguation": (36, 1),          # observed 0 on real AX210 HE CBR (not always 1)
}

# Codebook (feedback_type, codebook_info) -> (b_psi, b_phi). b_phi = b_psi + 2.
# 802.11-2020 Table 9-89a (VHT) / 802.11ax equivalent.
CODEBOOK: dict[tuple[int, int], tuple[int, int]] = {
    (FB_SU, 0): (2, 4),
    (FB_SU, 1): (4, 6),
    (FB_MU, 0): (5, 7),
    (FB_MU, 1): (7, 9),
}

# Ng decode (code -> grouping value).
VHT_NG = {0: 1, 1: 2, 2: 4}             # 3 reserved
HE_NG = {0: 4, 1: 16}

# Bandwidth code -> nominal MHz (informational).
BW_MHZ = {0: 20, 1: 40, 2: 80, 3: 160}

# Ns (number of reported, grouped subcarriers) tables (Wi-BFI tone plans).
HE_NS = {  # bw_code -> {Ng: Ns}
    0: {4: 64, 16: 20},
    1: {4: 122, 16: 32},
    2: {4: 250, 16: 64},
    3: {4: 500, 16: 160},
}
VHT_NS = {
    0: {1: 52, 2: 30, 4: 16},
    1: {1: 108, 2: 58, 4: 30},
    2: {1: 234, 2: 122, 4: 62},
    3: {1: 468, 2: 242, 4: 122},
}

# Per-subcarrier 10-angle order for Nr=4, Nc=2 (FROZEN; spec sec. 3.2).
# idx:  0      1      2      3      4      5      6      7      8      9
# name: phi11  phi21  phi31  psi21  psi31  psi41  phi22  phi32  psi32  psi42
# Slot identity: phi at {0,1,2,6,7}; psi at {3,4,5,8,9}.
PHI_SLOTS_4x2 = frozenset({0, 1, 2, 6, 7})


# --------------------------------------------------------------------------- #
# Angle order builder (general Givens rule; spec sec. 3.2)
# --------------------------------------------------------------------------- #
def angle_slots(Nr: int, Nc: int) -> list[bool]:
    """Return per-angle phi/psi identity list (True == phi) for (Nr, Nc).

    For column i = 1..min(Nc, Nr-1): emit phi_{i,i}..phi_{Nr-1,i} (Nr-i phis)
    then psi_{i+1,i}..psi_{Nr,i} (Nr-i psis). Total Na = 2*sum_{i}(Nr-i).
    For (4,2) this yields [phi,phi,phi,psi,psi,psi, phi,phi,psi,psi].
    """
    is_phi: list[bool] = []
    for i in range(1, min(Nc, Nr - 1) + 1):
        count = Nr - i
        is_phi.extend([True] * count)   # phi_{i,i}..phi_{Nr-1,i}
        is_phi.extend([False] * count)  # psi_{i+1,i}..psi_{Nr,i}
    return is_phi


def order_bits(Nr: int, Nc: int, b_psi: int, b_phi: int) -> list[int]:
    """Per-angle bit widths, in transmit order (spec sec. 3.2)."""
    return [b_phi if phi else b_psi for phi in angle_slots(Nr, Nc)]


# --------------------------------------------------------------------------- #
# Dequantiser / inverse quantiser (spec sec. 4) — mid-rise uniform quantiser
# --------------------------------------------------------------------------- #
def dequantise_phi(k: int, b_phi: int) -> float:
    """phi index -> radians. phi = (k + 0.5) * pi / 2^(b_phi-1), in [0, 2*pi)."""
    return PI * (1.0 / 2 ** b_phi + k * (1.0 / 2 ** (b_phi - 1)))


def dequantise_psi(k: int, b_psi: int) -> float:
    """psi index -> radians. psi = (k + 0.5) * (pi/2) / 2^b_psi, in [0, pi/2)."""
    return PI * (1.0 / 2 ** (b_psi + 2) + k * (1.0 / 2 ** (b_psi + 1)))


def quantise_phi(phi: float, b_phi: int) -> int:
    """radians -> phi index (inverse of :func:`dequantise_phi`)."""
    k = int(round(phi / PI * 2 ** (b_phi - 1) - 0.5))
    return int(np.clip(k, 0, 2 ** b_phi - 1))


def quantise_psi(psi: float, b_psi: int) -> int:
    """radians -> psi index (inverse of :func:`dequantise_psi`)."""
    k = int(round(psi / PI * 2 ** (b_psi + 1) - 0.5))
    return int(np.clip(k, 0, 2 ** b_psi - 1))


def _dequant(k: int, nbits: int, is_phi: bool) -> float:
    return dequantise_phi(k, nbits) if is_phi else dequantise_psi(k, nbits)


# --------------------------------------------------------------------------- #
# Bit reader / writer (LSB-first at both levels; spec sec. 3.3)
# --------------------------------------------------------------------------- #
def _bitstring_lsb(byte_seq: bytes) -> str:
    """Bytes -> bit string, each byte reversed so its LSB comes first."""
    return "".join(format(b, "08b")[::-1] for b in byte_seq)


def _read_angle(bitstr: str, pos: int, nbits: int) -> tuple[int, int]:
    """Read an ``nbits`` field; the first-transmitted bit is the value's LSB."""
    field = bitstr[pos:pos + nbits][::-1]
    return int(field, 2), pos + nbits


def _pack_bits_lsb(bitstr: str) -> bytes:
    """Inverse of :func:`_bitstring_lsb`: pad to a byte boundary, reverse each
    8-bit group, pack to bytes."""
    if len(bitstr) % 8:
        bitstr = bitstr + "0" * (8 - len(bitstr) % 8)
    out = bytearray()
    for i in range(0, len(bitstr), 8):
        chunk = bitstr[i:i + 8][::-1]
        out.append(int(chunk, 2))
    return bytes(out)


def _write_angle(k: int, nbits: int) -> str:
    """Inverse of :func:`_read_angle`: value -> field bits (LSB transmitted
    first), i.e. format then reverse so first-transmitted bit is the LSB."""
    return format(k & ((1 << nbits) - 1), "0{}b".format(nbits))[::-1]


def _bits(val: int, lo: int, width: int) -> int:
    return (val >> lo) & ((1 << width) - 1)


# --------------------------------------------------------------------------- #
# Frame identification (spec sec. 1)
# --------------------------------------------------------------------------- #
def identify_frame(raw_bytes: bytes) -> tuple[str, bytes] | None:
    """(mode, report_bytes) for a category-first CBF action body, else None.

    raw_bytes[0]=Category, raw_bytes[1]=Action, raw_bytes[2:]=MIMO Control +
    Compressed Beamforming Report.
    """
    if len(raw_bytes) < 2:
        return None
    category, action = raw_bytes[0], raw_bytes[1]
    if category == ACTION_CAT_HE:
        if action not in HE_ACTION_CODES:
            return None
        return "HE", raw_bytes[2:]
    if category == ACTION_CAT_VHT:
        if action != VHT_ACTION_COMPRESSED_BF:
            return None
        return "VHT", raw_bytes[2:]
    return None


# --------------------------------------------------------------------------- #
# MIMO Control parse (spec sec. 2)
# --------------------------------------------------------------------------- #
def parse_mimo_control(raw: bytes, kind: str) -> dict | None:
    """Parse the leading MIMO Control field of a CBF report.

    ``raw`` is the report (MIMO Control first). ``kind`` is "HE" or "VHT".
    Returns a params dict (Nc, Nr, bw_code, Ng, b_psi, b_phi, feedback_type,
    codebook_info, remaining_segments, first_segment, mc_len, mode, ...), or
    None on a malformed/short field or reserved Ng.
    """
    mc_len = MC_LEN[kind]
    if len(raw) < mc_len:
        return None
    mc = int.from_bytes(raw[0:mc_len], "little")
    bitmap = HE_MC_BITS if kind == "HE" else VHT_MC_BITS

    def g(name: str) -> int:
        lo, w = bitmap[name]
        return _bits(mc, lo, w)

    Nc = g("nc_index") + 1
    Nr = g("nr_index") + 1
    bw_code = g("bw")
    ng_code = g("ng")
    codebook_info = g("codebook_info")
    feedback_type = g("feedback_type")
    remaining_segments = g("remaining_segments")
    first_segment = g("first_segment")

    if kind == "HE":
        Ng = HE_NG.get(ng_code)
        disambiguation = g("disambiguation")
        ru_start = g("ru_start")
        ru_end = g("ru_end")
        sounding_token = g("sounding_token")
    else:
        Ng = VHT_NG.get(ng_code)        # code 3 -> None (reserved)
        disambiguation = None
        ru_start = ru_end = None
        sounding_token = g("sounding_token")

    if Ng is None:
        log.warning("reserved Ng code %d (kind=%s) mc=%s",
                    ng_code, kind, raw[0:mc_len].hex())
        return None

    cb = CODEBOOK.get((feedback_type, codebook_info))
    if cb is None:
        log.warning("unknown codebook fb=%d cb=%d mc=%s",
                    feedback_type, codebook_info, raw[0:mc_len].hex())
        return None
    b_psi, b_phi = cb

    return {
        "mode": kind,
        "mc_len": mc_len,
        "Nc": Nc,
        "Nr": Nr,
        "bw_code": bw_code,
        "ng_code": ng_code,
        "Ng": Ng,
        "codebook_info": codebook_info,
        "feedback_type": feedback_type,
        "b_psi": b_psi,
        "b_phi": b_phi,
        "remaining_segments": remaining_segments,
        "first_segment": first_segment,
        "sounding_token": sounding_token,
        "disambiguation": disambiguation,
        "ru_start": ru_start,
        "ru_end": ru_end,
    }


def ns_table(params: dict) -> int | None:
    """Standard Ns for (mode, bw_code, Ng), or None if not tabulated."""
    table = HE_NS if params["mode"] == "HE" else VHT_NS
    return table.get(params["bw_code"], {}).get(params["Ng"])


# --------------------------------------------------------------------------- #
# Report decode (spec sec. 3) — report bytes -> [Ns, Na] radians
# --------------------------------------------------------------------------- #
def decode_report(report_bytes: bytes, params: dict,
                  *, info: dict | None = None) -> np.ndarray | None:
    """Decode a CBF report (MIMO Control first) into [Ns, Na] angle radians.

    Skips the Nc Average-SNR bytes, then reads ``Ns`` subcarriers of the
    per-subcarrier angle order. ``Ns`` is recovered empirically from the
    available bit count (spec sec. 5.1) and cross-checked against the standard
    table. Returns None on truncation / gross mis-size / multi-segment.
    """
    mc_len = params["mc_len"]
    Nc, Nr = params["Nc"], params["Nr"]
    b_psi, b_phi = params["b_psi"], params["b_phi"]

    obits = order_bits(Nr, Nc, b_psi, b_phi)
    slots = angle_slots(Nr, Nc)
    Na = len(obits)
    bits_per_sc = sum(obits)

    # Skip Nc Average-SNR int8 bytes (not part of the feature vector).
    snr_off = mc_len
    angle_bytes = report_bytes[snr_off + Nc:]
    avail_bits = len(angle_bytes) * 8
    if avail_bits < bits_per_sc:
        log.warning("report too short: %d angle bits < %d/sc", avail_bits, bits_per_sc)
        return None

    Ns_observed = avail_bits // bits_per_sc
    Ns_tab = ns_table(params)
    # Prefer the standard table when it fits; otherwise use what the device emitted.
    if Ns_tab is not None and Ns_tab <= Ns_observed:
        Ns = Ns_tab
    else:
        Ns = Ns_observed
        if Ns_tab is not None:
            log.warning("Ns table=%s but only %d subcarriers fit (%d angle bits)",
                        Ns_tab, Ns_observed, avail_bits)

    need_bits = Ns * bits_per_sc
    # Slack check: trailing bytes should be <= padding (<8) + optional 4-byte FCS.
    slack = avail_bits - need_bits
    if slack < 0 or slack >= 8 + 32:
        log.warning("report mis-sized: avail=%d need=%d slack=%d", avail_bits, need_bits, slack)
        return None

    bitstr = _bitstring_lsb(angle_bytes)[:need_bits]

    angles = np.empty((Ns, Na), dtype=np.float32)
    pos = 0
    for s in range(Ns):
        for a, nbits in enumerate(obits):
            k, pos = _read_angle(bitstr, pos, nbits)
            angles[s, a] = _dequant(k, nbits, is_phi=slots[a])

    if info is not None:
        info["ns_raw"] = int(Ns)
        info["ns_table"] = Ns_tab
        info["bits_per_sc"] = int(bits_per_sc)
    return angles


# --------------------------------------------------------------------------- #
# Subcarrier mapping -> canonical 74 channels (spec sec. 5.2; single helper)
# --------------------------------------------------------------------------- #
def map_to_contract_channels(angles: np.ndarray,
                             info: dict | None = None) -> np.ndarray:
    """Map [Ns, Na] -> [NCH(=74), Na] deterministically.

    Ns == 74 -> pass through. Ns > 74 -> uniform linspace decimation. Ns < 74 ->
    zero-pad the tail. Always records ns_raw + the chosen mapping in ``info``
    when Ns != 74 (auditable). Swap only this body if the lab pins an exact
    74-index list later.
    """
    Ns = angles.shape[0]
    if info is None:
        info = {}
    if Ns == NCH:
        info.setdefault("subcarrier_map", "passthrough")
        return angles
    if Ns > NCH:
        idx = np.linspace(0, Ns - 1, NCH).round().astype(int)
        info["subcarrier_map"] = "decimate_linspace"
        info["ns_raw"] = int(Ns)
        return angles[idx]
    out = np.zeros((NCH, angles.shape[1]), dtype=angles.dtype)
    out[:Ns] = angles
    info["subcarrier_map"] = "zeropad"
    info["short_subcarriers"] = True
    info["ns_raw"] = int(Ns)
    return out


def flatten_to_features(angles_74: np.ndarray) -> np.ndarray:
    """[74, Na] -> length-740 float32, subcarrier-major C-order (spec sec. 5.3).

    feature[sc*Na + a] = angle a of subcarrier sc.
    """
    if angles_74.shape != (NCH, NA):
        raise ValueError(f"expected [{NCH},{NA}], got {angles_74.shape}")
    return np.ascontiguousarray(angles_74, dtype=np.float32).reshape(NFEAT)


# --------------------------------------------------------------------------- #
# Top-level frame decode: category-first body -> length-740 vector (or None)
# --------------------------------------------------------------------------- #
def decode_action_body(raw_bytes: bytes, *, info: dict | None = None,
                       require_target: bool = True) -> np.ndarray | None:
    """Decode one CBF action body -> length-740 float32 vector, or None.

    ``require_target`` enforces the lab target validation (Nc==2, Nr==4,
    HE Disambiguation==1, feedback_type==SU). Returns None on any mismatch,
    truncation, multi-segment, or unexpected error so the caller can skip the
    frame.
    """
    try:
        ident = identify_frame(raw_bytes)
        if ident is None:
            return None
        mode, report = ident
        params = parse_mimo_control(report, mode)
        if params is None:
            return None

        if require_target:
            if params["Nc"] != 2 or params["Nr"] != 4:
                log.warning("non-target geometry Nc=%d Nr=%d mc=%s",
                            params["Nc"], params["Nr"], report[:params["mc_len"]].hex())
                return None
            if params["feedback_type"] != FB_SU:
                log.warning("non-SU feedback_type=%d mc=%s",
                            params["feedback_type"], report[:params["mc_len"]].hex())
                return None
            # NB: the HE "disambiguation" bit is 0 on real AX210 CBRs (not the
            # once-assumed 1), so it is not a target gate. The rest of the MIMO
            # Control field is validated against real captures (2026-06-02).

        if params["remaining_segments"] != 0:
            log.warning("multi-segment report (remaining=%d); reassembly TODO",
                        params["remaining_segments"])
            return None

        angles = decode_report(report, params, info=info)
        if angles is None:
            return None
        angles_74 = map_to_contract_channels(angles, info=info)
        return flatten_to_features(angles_74)
    except Exception:  # noqa: BLE001 - any parse error -> skip frame
        log.exception("decode_action_body failed (raw=%s)", raw_bytes[:16].hex())
        return None


# --------------------------------------------------------------------------- #
# ENCODER (test-only inverse) — build valid CBF frames from angle indices
# --------------------------------------------------------------------------- #
def build_mimo_control(params: dict) -> bytes:
    """Pack a MIMO Control field from a params dict (inverse of
    :func:`parse_mimo_control`). Uses the named bit tables, so it stays exactly
    consistent with the decoder."""
    kind = params["mode"]
    mc_len = MC_LEN[kind]
    bitmap = HE_MC_BITS if kind == "HE" else VHT_MC_BITS

    # Reverse Ng value -> code.
    ng_map = HE_NG if kind == "HE" else VHT_NG
    ng_code = next(code for code, val in ng_map.items() if val == params["Ng"])

    fields = {
        "nc_index": params["Nc"] - 1,
        "nr_index": params["Nr"] - 1,
        "bw": params["bw_code"],
        "ng": ng_code,
        "codebook_info": params["codebook_info"],
        "feedback_type": params["feedback_type"],
        "remaining_segments": params.get("remaining_segments", 0),
        "first_segment": params.get("first_segment", 1),
        "sounding_token": params.get("sounding_token", 0),
    }
    if kind == "HE":
        fields["ru_start"] = params.get("ru_start", 0)
        fields["ru_end"] = params.get("ru_end", 0x3F)
        fields["disambiguation"] = params.get("disambiguation", 1)

    mc = 0
    for name, value in fields.items():
        lo, width = bitmap[name]
        mc |= (int(value) & ((1 << width) - 1)) << lo
    return mc.to_bytes(mc_len, "little")


def encode_report(indices: np.ndarray, params: dict,
                  *, snr: Iterable[int] | None = None) -> bytes:
    """Encode angle INDICES [Ns, Na] -> CBF report bytes (MIMO Control first).

    Exact inverse of :func:`decode_report`: writes MIMO Control, Nc Average-SNR
    int8 bytes, then the LSB-first angle bitstream. ``indices`` holds unsigned
    quantiser indices (NOT radians).
    """
    Nc, Nr = params["Nc"], params["Nr"]
    b_psi, b_phi = params["b_psi"], params["b_phi"]
    obits = order_bits(Nr, Nc, b_psi, b_phi)
    Ns, Na = indices.shape
    if Na != len(obits):
        raise ValueError(f"indices Na={Na} != order {len(obits)}")

    mc = build_mimo_control(params)
    snr_vals = list(snr) if snr is not None else [0] * Nc
    if len(snr_vals) != Nc:
        raise ValueError("snr must have Nc values")
    snr_bytes = bytes(int(v) & 0xFF for v in snr_vals)

    bit_parts: list[str] = []
    for s in range(Ns):
        for a, nbits in enumerate(obits):
            bit_parts.append(_write_angle(int(indices[s, a]), nbits))
    angle_bytes = _pack_bits_lsb("".join(bit_parts))
    return mc + snr_bytes + angle_bytes


def encode_action_body(indices: np.ndarray, params: dict,
                       *, snr: Iterable[int] | None = None,
                       action_code: int | None = None) -> bytes:
    """Encode a full category-first action body (Category, Action, report)."""
    kind = params["mode"]
    if kind == "HE":
        category = ACTION_CAT_HE
        action = action_code if action_code is not None else HE_ACTION_COMPRESSED_BF
    else:
        category = ACTION_CAT_VHT
        action = action_code if action_code is not None else VHT_ACTION_COMPRESSED_BF
    report = encode_report(indices, params, snr=snr)
    return bytes([category, action]) + report


def build_action_frame(indices: np.ndarray, params: dict, *,
                       addr1: str = "00:11:22:33:44:55",
                       addr2: str = "66:77:88:99:aa:bb",
                       addr3: str | None = None,
                       subtype: int | None = None,
                       snr: Iterable[int] | None = None,
                       action_code: int | None = None):
    """Build a scapy RadioTap/Dot11/Dot11Action CBF frame (test helper).

    Imports scapy lazily so this module stays numpy-only at import time. HE uses
    Action No Ack (subtype 14) by default; VHT uses Action (subtype 13).
    """
    from scapy.all import RadioTap, Dot11, Dot11Action, Raw  # local import

    kind = params["mode"]
    if subtype is None:
        subtype = 14 if kind == "HE" else 13
    if addr3 is None:
        addr3 = addr1
    if kind == "HE":
        category = ACTION_CAT_HE
        action = action_code if action_code is not None else HE_ACTION_COMPRESSED_BF
    else:
        category = ACTION_CAT_VHT
        action = action_code if action_code is not None else VHT_ACTION_COMPRESSED_BF

    report = encode_report(indices, params, snr=snr)
    load = bytes([action]) + report
    return (RadioTap()
            / Dot11(type=0, subtype=subtype, addr1=addr1, addr2=addr2, addr3=addr3)
            / Dot11Action(category=category)
            / Raw(load=load))
