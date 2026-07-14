"""Round-trip tests for the CSI decoder.

Proves the real FeitCSI binary decoder bit-for-bit with no hardware (encode
real FeitCSI records -> decode -> exact [T,212]+dt recovery, documented flatten
order, dt[0]==0), the >53-subcarrier linspace decimation, and that ``parse_csi``
raises on an undecodable ``.raw`` (no placeholder path). numpy only.

Run:  .venv/bin/python parsers/tests/test_csi_decode.py
  or: .venv/bin/python -m pytest parsers/tests/test_csi_decode.py -q
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

# Allow running as a plain script (python parsers/tests/test_csi_decode.py).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from wallflower import contract
from parsers import csi_decode
from parsers import parse_csi

NSUB = contract.CSI_SUBCARRIERS   # 53
NANT = contract.CSI_ANTENNAS      # 2
NCOMP = contract.CSI_COMPONENTS   # 2
NFEAT = contract.CSI_FEATURES     # 212


# --------------------------------------------------------------------------- #
# features_from_csi flatten order is exactly feature[sc*4 + ant*2 + comp].
# --------------------------------------------------------------------------- #
def test_flatten_order():
    rng = np.random.default_rng(0)
    H = (rng.integers(-100, 100, size=(NSUB, NANT))
         + 1j * rng.integers(-100, 100, size=(NSUB, NANT))).astype(np.complex64)
    feat = csi_decode.features_from_csi(H)
    assert feat.shape == (NFEAT,)
    assert feat.dtype == np.float32
    mag = np.abs(H)
    phase = np.angle(H)
    for sc in range(NSUB):
        for ant in range(NANT):
            assert abs(feat[sc * 4 + ant * 2 + 0] - mag[sc, ant]) < 1e-5
            assert abs(feat[sc * 4 + ant * 2 + 1] - phase[sc, ant]) < 1e-5
    print("test_flatten_order OK")


# --------------------------------------------------------------------------- #
# FeitCSI binary round trip: real records -> decode -> [T,212]+dt.
# --------------------------------------------------------------------------- #
def test_feitcsi_roundtrip():
    name = "P001_normal_001_csi_p1"
    blob = csi_decode.make_feitcsi_blocks(name, num_rx=2, num_tx=1, num_sub=NSUB)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "csi_p1.raw"
        raw.write_bytes(blob)

        info: dict = {}
        out = csi_decode.decode_feitcsi(raw, info=info)
        assert out is not None, "FeitCSI decode returned None"
        x, dt = out

        assert x.ndim == 2 and x.shape[1] == NFEAT, x.shape
        assert x.shape[1] == contract.CSI_FEATURES
        assert x.dtype == np.float32
        assert dt.shape == (x.shape[0],)
        assert dt.dtype == np.float32
        assert float(dt[0]) == 0.0
        assert np.all(dt >= 0.0)
        assert x.shape[0] >= 2
        assert info["subcarrier_map"] == "passthrough"

        # Round-trip the FIRST block bit-for-bit against features_from_csi so the
        # documented flatten order (sc-major, ant, comp) is asserted on real bytes.
        blocks = csi_decode.read_feitcsi_blocks(raw)
        h0, H0 = blocks[0]
        H2 = csi_decode.select_antennas(H0)
        H53 = csi_decode.select_subcarriers(H2)
        expect0 = csi_decode.features_from_csi(H53)
        assert np.allclose(x[0], expect0), "first-packet features must round-trip"

    # Median rate sanity vs contract nominal (~285 Hz); wide tolerance.
    assert info.get("median_rate_hz") is not None
    print(f"test_feitcsi_roundtrip OK (T={x.shape[0]}, "
          f"median_rate_hz={info['median_rate_hz']:.1f})")


# --------------------------------------------------------------------------- #
# >53 subcarriers (e.g. 160 MHz) decimates to 53 via linspace, records ns_raw.
# --------------------------------------------------------------------------- #
def test_feitcsi_decimate():
    name = "P001_normal_001_csi_p2"
    blob = csi_decode.make_feitcsi_blocks(name, num_rx=2, num_tx=1, num_sub=1992,
                                          seconds=1.0)
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "csi_p2.raw"
        raw.write_bytes(blob)
        info: dict = {}
        out = csi_decode.decode_feitcsi(raw, info=info)
    assert out is not None
    x, dt = out
    assert x.shape[1] == NFEAT
    assert info["subcarrier_map"] == "decimate_linspace"
    assert info["ns_raw"] == 1992
    assert float(dt[0]) == 0.0
    print("test_feitcsi_decimate OK (1992 -> 53 linspace)")


# --------------------------------------------------------------------------- #
# parse_csi on an empty/undecodable .raw -> RuntimeError (no placeholder path).
# --------------------------------------------------------------------------- #
def test_parse_csi_undecodable_raises():
    with tempfile.TemporaryDirectory() as td:
        data_root = Path(td) / "data"
        raw = Path(td) / "csi_p1.raw"
        raw.write_bytes(b"")  # nothing decodable
        try:
            parse_csi.parse_csi(raw, perspective=1, participant="P001",
                                style="normal", trial="001", data_root=data_root)
        except RuntimeError:
            print("test_parse_csi_undecodable_raises OK (no placeholder fallback)")
            return
        raise AssertionError("expected RuntimeError on an undecodable CSI .raw")


# --------------------------------------------------------------------------- #
# parse_csi on a real FeitCSI .raw -> real decode, 212 schema.
# --------------------------------------------------------------------------- #
def test_parse_csi_feitcsi_real():
    name = contract.sample_basename("P001", "normal", "001", "csi", 1)
    blob = csi_decode.make_feitcsi_blocks(name, num_rx=2, num_tx=1, num_sub=NSUB)
    with tempfile.TemporaryDirectory() as td:
        data_root = Path(td) / "data"
        raw = Path(td) / "csi_p1.raw"
        raw.write_bytes(blob)
        info = parse_csi.parse_csi(raw, perspective=1, participant="P001",
                                   style="normal", trial="001",
                                   data_root=data_root)
        assert info["n_features"] == contract.CSI_FEATURES
        with np.load(Path(info["path"]), allow_pickle=True) as z:
            assert z["x"].shape[1] == contract.CSI_FEATURES
            assert float(z["dt"][0]) == 0.0
    print("test_parse_csi_feitcsi_real OK (real decode)")


def main() -> int:
    test_flatten_order()
    test_feitcsi_roundtrip()
    test_feitcsi_decimate()
    test_parse_csi_undecodable_raises()
    test_parse_csi_feitcsi_real()
    print("\nALL CSI DECODE TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
