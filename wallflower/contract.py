"""Canonical, single-source-of-truth constants and helpers for the whole lab.

Every module (orchestrator, nodes, capture, parsers, models) imports from here
so that file naming, directory layout, metadata fields, modality feature
dimensions and label vocabularies never drift between components.

All numeric/feature constants are taken directly from the BFId paper:
  * BFI : 740 features = 10 quantised angles x 74 channels  (+1 time-delta)
  * CSI : 212 features = (phase + magnitude) x 53 subcarriers x 2 antennas
          i.e. 53 * 2 * 2 = 212                              (+1 time-delta)
  * Sample rates (empirical, paper sec. 5.1.2): ~10 Hz BFI, ~285 Hz CSI
  * Walking styles: normal (x20 back-and-forth), backpack/crate/fast/turnstile (x10)
  * 4 perspectives, 2x AX210 radios per perspective node (radio A=CSI, B=BFI)
  * Operative pilot RF: 5 GHz, 80 MHz, channel 36 (single ASUS AP)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Modalities
# --------------------------------------------------------------------------- #
Modality = Literal["bfi", "csi"]
MODALITIES: tuple[Modality, ...] = ("bfi", "csi")

# Per the paper. The +1 is the appended time-delta (dt) channel that the models
# consume as an extra temporal feature; parsers store dt separately in the npz
# under "dt" and the dataset layer concatenates it as the final input column.
BFI_FEATURES = 740          # 10 quantised angles x 74 channels
BFI_ANGLES = 10
BFI_CHANNELS = 74
CSI_FEATURES = 212          # (phase+mag) x 53 subcarriers x 2 antennas
CSI_SUBCARRIERS = 53
CSI_ANTENNAS = 2
CSI_COMPONENTS = 2          # phase, magnitude

FEATURE_DIMS: dict[str, int] = {"bfi": BFI_FEATURES, "csi": CSI_FEATURES}

# Nominal empirically-tuned sample rates (Hz). A sanity reference when
# validating real captures.
NOMINAL_RATE_HZ: dict[str, float] = {"bfi": 10.0, "csi": 285.0}

# --------------------------------------------------------------------------- #
# Labels / experimental factors
# --------------------------------------------------------------------------- #
WALKING_STYLES: tuple[str, ...] = ("normal", "backpack", "crate", "fast", "turnstile")

# Paper protocol: normal walking repeated 20x back-and-forth, others 10x.
STYLE_REPEATS: dict[str, int] = {
    "normal": 20,
    "backpack": 10,
    "crate": 10,
    "fast": 10,
    "turnstile": 10,
}

DIRECTIONS: tuple[str, ...] = ("forward", "return")  # back-and-forth legs
PERSPECTIVES: tuple[int, ...] = (1, 2, 3, 4)

# --------------------------------------------------------------------------- #
# RF / topology constants
# --------------------------------------------------------------------------- #
BAND_GHZ = 5
WIDTH_MHZ = 80
AP_CSI_CHANNEL = 36
AP_BFI_CHANNEL = 36
AP_CHANNELS: dict[str, int] = {"AP-CSI": AP_CSI_CHANNEL, "AP-BFI": AP_BFI_CHANNEL}

# Logical radio roles on a perspective node (each node has 2x AX210).
RADIO_ROLES = ("csi", "bfi")  # radio A -> csi monitor/capture, radio B -> bfi client


# Operative 5 GHz width (the ASUS AP runs ch 36 / 80 MHz). Kept separately for
# readability at call sites.
FIVEGHZ_WIDTH_MHZ = 80


# --- 6 GHz channel <-> frequency mapping ------------------------------------ #
# On 6 GHz, `iw ... set channel <n>` is ambiguous with legacy 2.4/5 GHz channel
# numbering and fails ("Unknown channel"). 6 GHz must be tuned by FREQUENCY:
#   iw dev <if> set freq <control_mhz> 160MHz <center_mhz>
# Channel center frequency: freq = 5950 + 5*channel (the global 6 GHz grid).
def sixghz_control_freq(channel: int) -> int:
    """Control-channel centre frequency (MHz) for a 6 GHz channel number."""
    return 5950 + 5 * channel


def sixghz_160_center_freq(channel: int) -> int:
    """Centre frequency (MHz) of the 160 MHz block containing this 6 GHz channel.

    160 MHz blocks are 8x 20 MHz channels wide, starting at ch 1 (blocks begin at
    ch 1, 33, 65, ...); the block centre is 14 channel-units above its start.
    e.g. ch 37 -> 6185 MHz, ch 85 -> 6345 MHz (control 6135 / 6375 MHz).
    """
    block_start = ((channel - 1) // 32) * 32 + 1
    center_channel = block_start + 14
    return 5950 + 5 * center_channel


# --- 5 GHz channel <-> frequency mapping ------------------------------------ #
# Legacy 5 GHz grid: freq = 5000 + 5*channel (ch 36 -> 5180 MHz). Unlike 6 GHz,
# `iw ... set channel 36` works here, but we tune by FREQUENCY for both bands so
# the monitor bring-up path is uniform (and so an explicit 80 MHz centre is set).
def fiveghz_control_freq(channel: int) -> int:
    """Control-channel centre frequency (MHz) for a 5 GHz channel number."""
    return 5000 + 5 * channel


def fiveghz_80_center_freq(channel: int) -> int:
    """Centre frequency (MHz) of the 80 MHz block containing this 5 GHz channel.

    80 MHz blocks group four 20 MHz channels (e.g. 36/40/44/48). For the lower
    5 GHz channels (UNII-1/2/2e, multiples of 4 from ch 36) the blocks align on a
    16-channel-unit grid from ch 36, so the block centre sits 6 channel-units
    above the block start: ch 36 -> block 36..48 -> center ch 42 -> 5210 MHz.
    NOTE: UNII-3 (ch 149+) uses a different alignment and is not handled here;
    the lab only captures on ch 36.
    """
    block_start = ((channel - 36) // 16) * 16 + 36
    center_channel = block_start + 6
    return 5000 + 5 * center_channel


def monitor_freq_args(channel: int, band_ghz: int = BAND_GHZ) -> list[str]:
    """`iw ... set freq` arguments to tune a monitor VIF to ``channel``.

    Band-aware. Uses iw's NUMERIC width form with an explicit centre frequency
    in both bands:
        iw dev <if> set freq <control> <width> <center>
    (the keyword form `... 160MHz <center>` is rejected by iw, and bare
    `set channel <n>` fails on 6 GHz).

    * ``band_ghz=6`` (the paper's nominal plan): 160 MHz blocks; e.g. ch 85
      -> ['6375', '160', '6345'].
    * ``band_ghz=5`` (the operative lab reality — ASUS on ch 36 / 80 MHz):
      80 MHz blocks; e.g. ch 36 -> ['5180', '80', '5210'].

    A bare channel number is ambiguous across bands (ch 36 exists in both), so
    the band must be supplied; it defaults to the operative BAND_GHZ.
    """
    if int(band_ghz) == 5:
        return [str(fiveghz_control_freq(channel)),
                str(FIVEGHZ_WIDTH_MHZ),
                str(fiveghz_80_center_freq(channel))]
    return [str(sixghz_control_freq(channel)),
            "160",
            str(sixghz_160_center_freq(channel))]

# --------------------------------------------------------------------------- #
# Identifier validation
# --------------------------------------------------------------------------- #
PARTICIPANT_RE = re.compile(r"^P\d{3,}$")     # e.g. P001
TRIAL_RE = re.compile(r"^\d{3,}$")            # e.g. 001


def validate_participant(pid: str) -> str:
    if not PARTICIPANT_RE.match(pid):
        raise ValueError(f"participant id {pid!r} must match {PARTICIPANT_RE.pattern}")
    return pid


def validate_style(style: str) -> str:
    if style not in WALKING_STYLES:
        raise ValueError(f"style {style!r} not in {WALKING_STYLES}")
    return style


def validate_trial(trial: str) -> str:
    if not TRIAL_RE.match(trial):
        raise ValueError(f"trial id {trial!r} must match {TRIAL_RE.pattern}")
    return trial


def validate_perspective(p: int) -> int:
    if p not in PERSPECTIVES:
        raise ValueError(f"perspective {p!r} not in {PERSPECTIVES}")
    return p


# --------------------------------------------------------------------------- #
# Canonical filesystem layout
# --------------------------------------------------------------------------- #
# data/raw/participant=P001/style=normal/trial=001/{metadata.json,csi_p1.raw,
#          ...,bfi_recorder.pcapng,logs/}
# data/processed/{samples.parquet, samples/<sample_name>.npz}

def raw_trial_dir(data_root: Path | str, participant: str, style: str, trial: str) -> Path:
    """Hive-partitioned raw directory for one trial."""
    return (
        Path(data_root)
        / "raw"
        / f"participant={validate_participant(participant)}"
        / f"style={validate_style(style)}"
        / f"trial={validate_trial(trial)}"
    )


def csi_raw_name(perspective: int) -> str:
    return f"csi_p{validate_perspective(perspective)}.raw"


def bfi_recorder_name() -> str:
    # The passive recorder captures *all* perspectives centrally in one pcapng.
    return "bfi_recorder.pcapng"


def metadata_name() -> str:
    return "metadata.json"


def sample_basename(participant: str, style: str, trial: str,
                    modality: Modality, perspective: int) -> str:
    """e.g. P001_normal_001_bfi_p1  (no extension)."""
    return (f"{validate_participant(participant)}_{validate_style(style)}_"
            f"{validate_trial(trial)}_{modality}_p{validate_perspective(perspective)}")


def processed_sample_path(data_root: Path | str, participant: str, style: str,
                          trial: str, modality: Modality, perspective: int) -> Path:
    base = sample_basename(participant, style, trial, modality, perspective)
    return Path(data_root) / "processed" / "samples" / f"{base}.npz"


SAMPLES_PARQUET = "samples.parquet"


# --------------------------------------------------------------------------- #
# .npz sample schema (parser output -> model input)
# --------------------------------------------------------------------------- #
# Each .npz holds one variable-length atomic sample:
#   x      : float32 [timesteps, features]   (features == FEATURE_DIMS[modality])
#   dt     : float32 [timesteps]             (seconds since previous data point;
#                                             dt[0] == 0.0)
#   label  : participant_id (str scalar)
#   style  : walking_style  (str scalar)
#   perspective : int 1..4
#   modality    : "bfi" | "csi"
NPZ_KEYS = ("x", "dt", "label", "style", "perspective", "modality")


def expected_feature_dim(modality: Modality, with_dt: bool = False) -> int:
    """Feature width of the model input. with_dt adds the appended dt column."""
    return FEATURE_DIMS[modality] + (1 if with_dt else 0)


# --------------------------------------------------------------------------- #
# Canonical metadata.json record (one per trial)
# --------------------------------------------------------------------------- #
@dataclass
class RadioRecord:
    """Physical radio -> logical role mapping captured at record time."""
    role: str                 # "csi" | "bfi" | "bfi_recorder" | "csi_traffic"
    node: str                 # logical node id, e.g. "node1"
    iface: str                # e.g. "wlp1s0"
    phy: str                  # e.g. "phy0"
    mac: str                  # e.g. "AA:BB:CC:DD:EE:02"
    pci: str = ""             # e.g. "01:00.0"
    perspective: int | None = None
    ap: str = ""              # "AP-CSI" | "AP-BFI" | ""
    channel: int | None = None


@dataclass
class TrialMetadata:
    """The one canonical metadata.json the orchestrator writes per trial."""
    schema_version: str
    participant: str
    trial: str
    style: str
    direction: str
    perspectives: list[int]
    timestamp_utc: str                       # ISO-8601, recording start
    band_ghz: int = BAND_GHZ
    width_mhz: int = WIDTH_MHZ
    ap_channels: dict[str, int] = field(default_factory=lambda: dict(AP_CHANNELS))
    radios: list[RadioRecord] = field(default_factory=list)
    clock_sync: dict = field(default_factory=dict)   # see orchestrator clock check
    notes: str = ""
    # Shared wall-clock at which the orchestrator fanned out the (concurrent)
    # CSI+BFI capture for this trial — the alignment anchor for the two
    # modalities recorded in ONE session. Unix epoch seconds (float) + ISO form.
    capture_start_epoch: float | None = None
    capture_start_utc: str = ""
    # Per-modality capture role/radio/channel summary + honest documentation of
    # what this node's radios can / cannot do simultaneously (Research B).
    capture_plan: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


METADATA_SCHEMA_VERSION = "1.0"
