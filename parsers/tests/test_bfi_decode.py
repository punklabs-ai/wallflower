"""Round-trip + crafted-pcap tests for the BFI compressed-beamforming decoder.

Proves bit-exact correctness of ``parsers.bfi_decode`` (encode -> bytes ->
decode -> exact angle-INDEX recovery, dequantised values within one step) and
that ``parsers.parse_bfi.decode_bfi`` assembles a [T,740]+dt series through a
crafted pcapng. No hardware required.

Run:  .venv/bin/python -m pytest parsers/tests/test_bfi_decode.py -q
  or: .venv/bin/python parsers/tests/test_bfi_decode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script (python parsers/tests/test_bfi_decode.py).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wallflower import contract
from parsers import bfi_decode
from parsers import parse_bfi

NA = contract.BFI_ANGLES   # 10
NCH = contract.BFI_CHANNELS  # 74
NFEAT = contract.BFI_FEATURES  # 740


def _skip(reason: str):
    """Skip via pytest if present, else raise to mark unsupported in script mode."""
    try:
        import pytest
        pytest.skip(reason)
    except ImportError:
        raise RuntimeError(f"cannot run test (skip): {reason}")


def _base_params(mode: str, codebook_info: int, Ng: int) -> dict:
    b_psi, b_phi = bfi_decode.CODEBOOK[(bfi_decode.FB_SU, codebook_info)]
    return {
        "mode": mode,
        "mc_len": bfi_decode.MC_LEN[mode],
        "Nc": 2,
        "Nr": 4,
        "bw_code": 3,             # 160 MHz
        "ng_code": 0,
        "Ng": Ng,
        "codebook_info": codebook_info,
        "feedback_type": bfi_decode.FB_SU,
        "b_psi": b_psi,
        "b_phi": b_phi,
        "remaining_segments": 0,
        "first_segment": 1,
        "sounding_token": 7,
        "disambiguation": 1 if mode == "HE" else None,
        "ru_start": 0,
        "ru_end": 0x3F,
    }


def _random_indices(Ns: int, params: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obits = bfi_decode.order_bits(params["Nr"], params["Nc"],
                                  params["b_psi"], params["b_phi"])
    idx = np.empty((Ns, len(obits)), dtype=np.int64)
    for a, nbits in enumerate(obits):
        idx[:, a] = rng.integers(0, 2 ** nbits, size=Ns)
    return idx


def _recover_indices(angles: np.ndarray, params: dict) -> np.ndarray:
    """Re-quantise decoded radians back to indices using the inverse quantiser."""
    slots = bfi_decode.angle_slots(params["Nr"], params["Nc"])
    out = np.empty(angles.shape, dtype=np.int64)
    for a, is_phi in enumerate(slots):
        for s in range(angles.shape[0]):
            if is_phi:
                out[s, a] = bfi_decode.quantise_phi(angles[s, a], params["b_phi"])
            else:
                out[s, a] = bfi_decode.quantise_psi(angles[s, a], params["b_psi"])
    return out


# --------------------------------------------------------------------------- #
# Pure-bit round trip: index recovery is EXACT.
# --------------------------------------------------------------------------- #
def _roundtrip_indices(mode: str, codebook_info: int, Ns: int, Ng: int, seed: int):
    params = _base_params(mode, codebook_info, Ng)
    idx = _random_indices(Ns, params, seed)

    report = bfi_decode.encode_report(idx, params, snr=[5, -3])

    # MIMO Control round-trips.
    parsed = bfi_decode.parse_mimo_control(report, mode)
    assert parsed is not None
    for k in ("Nc", "Nr", "bw_code", "Ng", "codebook_info", "feedback_type",
              "b_psi", "b_phi"):
        assert parsed[k] == params[k], (k, parsed[k], params[k])
    if mode == "HE":
        assert parsed["disambiguation"] == 1

    # Decode report and recover indices exactly.
    info: dict = {}
    angles = bfi_decode.decode_report(report, parsed, info=info)
    assert angles is not None
    assert angles.shape == (Ns, NA)
    rec = _recover_indices(angles, params)
    assert np.array_equal(rec, idx), "index recovery must be bit-exact"

    # Dequantised values equal _dequant(K, ...) exactly.
    slots = bfi_decode.angle_slots(params["Nr"], params["Nc"])
    for a, is_phi in enumerate(slots):
        nbits = params["b_phi"] if is_phi else params["b_psi"]
        for s in range(Ns):
            expect = bfi_decode._dequant(int(idx[s, a]), nbits, is_phi)
            assert abs(float(angles[s, a]) - expect) < 1e-5
    return params, idx, info


def test_roundtrip_he_codebook0():
    _roundtrip_indices("HE", 0, Ns=74, Ng=16, seed=1)


def test_roundtrip_he_codebook1():
    _roundtrip_indices("HE", 1, Ns=74, Ng=16, seed=2)


def test_roundtrip_vht_codebook0():
    _roundtrip_indices("VHT", 0, Ns=74, Ng=4, seed=3)


def test_roundtrip_vht_codebook1():
    _roundtrip_indices("VHT", 1, Ns=74, Ng=4, seed=4)


def test_value_within_one_step():
    """Encode->decode error vs the originally chosen radians < one step."""
    params = _base_params("HE", 0, 16)
    rng = np.random.default_rng(11)
    Ns = 74
    slots = bfi_decode.angle_slots(params["Nr"], params["Nc"])
    # Choose random radians, quantise to indices, encode, decode.
    chosen = np.empty((Ns, NA), dtype=np.float64)
    idx = np.empty((Ns, NA), dtype=np.int64)
    for a, is_phi in enumerate(slots):
        if is_phi:
            chosen[:, a] = rng.uniform(0, 2 * np.pi, Ns)
            idx[:, a] = [bfi_decode.quantise_phi(v, params["b_phi"]) for v in chosen[:, a]]
        else:
            chosen[:, a] = rng.uniform(0, np.pi / 2, Ns)
            idx[:, a] = [bfi_decode.quantise_psi(v, params["b_psi"]) for v in chosen[:, a]]
    report = bfi_decode.encode_report(idx, params)
    parsed = bfi_decode.parse_mimo_control(report, "HE")
    angles = bfi_decode.decode_report(report, parsed)
    step_phi = np.pi / 2 ** (params["b_phi"] - 1)
    step_psi = (np.pi / 2) / 2 ** params["b_psi"]
    for a, is_phi in enumerate(slots):
        step = step_phi if is_phi else step_psi
        # Skip samples that clipped at the range edge (no valid bound there).
        for s in range(Ns):
            kmax = (2 ** params["b_phi"] - 1) if is_phi else (2 ** params["b_psi"] - 1)
            if idx[s, a] in (0, kmax):
                continue
            assert abs(float(angles[s, a]) - chosen[s, a]) < step


# --------------------------------------------------------------------------- #
# Through _decode_action_frame (full body) + flatten order.
# --------------------------------------------------------------------------- #
def test_decode_action_frame_passthrough_74():
    params = _base_params("HE", 0, 16)
    idx = _random_indices(74, params, seed=5)
    body = bfi_decode.encode_action_body(idx, params)
    vec = parse_bfi._decode_action_frame(body)
    assert vec is not None
    assert vec.shape == (NFEAT,)
    assert vec.dtype == np.float32
    # Subcarrier-major flatten: feature[sc*10 + a] == dequant(idx[sc,a]).
    slots = bfi_decode.angle_slots(4, 2)
    for sc in range(74):
        for a, is_phi in enumerate(slots):
            nbits = params["b_phi"] if is_phi else params["b_psi"]
            expect = bfi_decode._dequant(int(idx[sc, a]), nbits, is_phi)
            assert abs(float(vec[sc * NA + a]) - expect) < 1e-5


def test_decode_action_frame_decimate_160():
    """Ns in {160,500} decimates to exactly 74, records ns_raw."""
    for Ns in (160, 500):
        params = _base_params("HE", 0, 16 if Ns == 160 else 4)
        idx = _random_indices(Ns, params, seed=Ns)
        report = bfi_decode.encode_report(idx, params)
        parsed = bfi_decode.parse_mimo_control(report, "HE")
        info: dict = {}
        angles = bfi_decode.decode_report(report, parsed, info=info)
        assert angles.shape[0] == Ns
        mapped = bfi_decode.map_to_contract_channels(angles, info=info)
        assert mapped.shape == (NCH, NA)
        assert info["subcarrier_map"] == "decimate_linspace"
        assert info["ns_raw"] == Ns


def test_non_target_geometry_returns_none():
    # Nc=1 is a valid CBF frame but not the lab target -> decoder rejects it.
    params = _base_params("HE", 0, 16)
    params["Nc"] = 1
    idx = _random_indices(74, params, seed=7)  # Nc=1 -> 3 angles/sc
    body = bfi_decode.encode_action_body(idx, params)
    assert parse_bfi._decode_action_frame(body) is None


def test_non_cbf_frame_returns_none():
    assert parse_bfi._decode_action_frame(bytes([0, 0, 1, 2, 3])) is None
    assert parse_bfi._decode_action_frame(bytes([21, 99])) is None  # wrong VHT action


# --------------------------------------------------------------------------- #
# Crafted-pcap trial assembly through decode_bfi (requires scapy).
# --------------------------------------------------------------------------- #
def test_decode_bfi_trial_assembly(tmp_path):
    if not parse_bfi._HAVE_SCAPY:
        _skip("scapy unavailable")
    from scapy.all import wrpcap

    params = _base_params("HE", 0, 16)
    pkts = []
    # Out-of-order timestamps to verify chronological sort.
    times = [100.25, 100.0, 100.1]
    for i, t in enumerate(times):
        idx = _random_indices(74, params, seed=100 + i)
        pkt = bfi_decode.build_action_frame(idx, params)
        pkt.time = t
        pkts.append(pkt)
    pcap = tmp_path / "trial.pcapng"
    wrpcap(str(pcap), pkts)

    out = parse_bfi.decode_bfi(pcap, perspective=1)
    assert out is not None
    x, dt = out
    assert x.shape == (3, NFEAT)
    assert dt.shape == (3,)
    assert dt[0] == 0.0
    assert np.all(dt[1:] > 0)  # sorted -> strictly increasing timestamps
    # _validate_sample must accept the decoded sample.
    parse_bfi._validate_sample(x, dt)


def test_empty_pcap_raises(tmp_path):
    """parse_bfi on a pcap with no BFI frames raises (no placeholder path)."""
    if not parse_bfi._HAVE_SCAPY:
        _skip("scapy unavailable")
    from scapy.all import wrpcap, RadioTap

    pcap = tmp_path / "empty.pcapng"
    wrpcap(str(pcap), [RadioTap()])  # no CBF frames
    try:
        parse_bfi.parse_bfi(pcap, perspective=1, participant="P001",
                            style="normal", trial="001",
                            data_root=str(tmp_path / "data"))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError on a pcap with no BFI frames")


# --------------------------------------------------------------------------- #
# Script entrypoint (no pytest required).
# --------------------------------------------------------------------------- #
def _run_as_script() -> int:
    import logging
    import tempfile
    # Expected cross-check warnings (table Ns vs encoded Ns) are not failures.
    logging.disable(logging.WARNING)
    test_roundtrip_he_codebook0()
    test_roundtrip_he_codebook1()
    test_roundtrip_vht_codebook0()
    test_roundtrip_vht_codebook1()
    test_value_within_one_step()
    test_decode_action_frame_passthrough_74()
    test_decode_action_frame_decimate_160()
    test_non_target_geometry_returns_none()
    test_non_cbf_frame_returns_none()
    if parse_bfi._HAVE_SCAPY:
        with tempfile.TemporaryDirectory() as d:
            test_decode_bfi_trial_assembly(Path(d))
        with tempfile.TemporaryDirectory() as d:
            test_empty_pcap_raises(Path(d))
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(_run_as_script())
