"""viz.live_motion — live WiFi motion dashboard with noise reduction + logging.

Turns the per-frame RSSI of the associated 5 GHz link into a live,
browser-viewable motion view, and logs every processed sample to JSONL so
detection can be tuned offline (see analyze_motion.py).

Signal path (all numpy, no scipy):
  frames --> resample @FS (mean RSSI per bin) --> band-pass (drift + quantisation
  rejected) --> motion energy (RMS in motion window) --> threshold above a frozen
  still-baseline floor --> hysteresis/debounce.

Why this reduces false positives at rest:
  * consistent source: only frames transmitted BY the AP (stable RSSI reference),
  * band-pass keeps the human-motion band (~0.3-3 Hz) and drops 1-dBm quantisation
    jitter (broadband) and slow AGC/thermal drift,
  * the high-pass is mean-removal over the analysis window; the noise floor is a
    still-baseline locked over `calib_s` of (assumed-still) data then FROZEN
    (recalibratable via /calibrate), so the threshold sits just above the *actual*
    resting noise instead of a fixed guess, and hysteresis stops single-sample
    flicker from tripping presence.

Source is pluggable: real RSSI or fused CSI+BFI feature streams.

Run (venv has numpy):
    .venv/bin/python -m viz.live_motion --client-iface wlp1s0 --mon-iface mon1 \
        --gateway 198.51.100.1
The AP BSSID is auto-detected from the client interface's association by default.
open http://<host>:8088/ , then use the Still/Moving buttons to label ground truth.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from wallflower import contract

SIG_RE = re.compile(r"(-?\d+)dBm signal")

log = logging.getLogger("viz.live_motion")


class Cfg:
    """Tunable parameters (CLI-overridable) — the knobs we iterate on."""
    fs = 20.0               # resample rate (Hz)
    smooth_n = 3            # low-pass taps: rejects > ~fs/(2*smooth_n) Hz (quantisation)
    motion_win_s = 0.75     # window for motion energy (RMS)
    k = 3.5                 # threshold = floor + k * MAD  (sensitivity)
    floor_offset = 0.0      # manual offset added to the calibrated motion floor
    deb_on = 4              # consecutive samples above thr to assert presence
    deb_off = 10            # consecutive samples below thr to clear presence
    calib_s = 20.0          # seconds of (assumed-still) data to lock the floor
    p_margin = 1.12         # threshold is also kept >= resting p99 * this (calm-window guard)
    fftn = 64               # STFT window (heatmap)
    heat_decay = 0.985      # rolling heatmap scale decay; lower adapts faster
    heat_floor = 0.18       # hide low-level resting texture in the heatmap


cfg = Cfg()


class RouterDSP:
    """Per-router motion pipeline — the identical path every router runs:
    resample -> band-pass -> motion energy (RMS) -> frozen still-baseline
    threshold -> hysteresis. One instance per router; Router 1, 2, and 3 are
    processed the same way, each from its own real capture.
    """

    def __init__(self) -> None:
        self.xbuf: deque[float] = deque([np.nan] * cfg.fftn, maxlen=cfg.fftn)
        self.warmup = int(2.0 * cfg.fs)
        self.calib_target = max(20, int(cfg.calib_s * cfg.fs))
        self.calib_buf: list[float] = []
        self.calibrating = True
        self.floor = self.mad = self.p_hi = 0.0
        self.last = -50.0
        self.on_count = self.off_count = 0
        self.presence = False
        self.i = 0
        self.calib_request = False
        self.last_bp = np.zeros(cfg.fftn)

    def request_calibration(self) -> None:
        self.calibrating = True
        self.calib_buf = []

    def step(self, recent: list[float]) -> dict:
        """Advance one tick from this router's recent RSSI frames."""
        FS = cfg.fs
        FFTN = cfg.fftn
        samp = float(np.mean(recent)) if recent else self.last
        self.last = samp
        self.xbuf.append(samp)

        smooth_n = max(1, int(cfg.smooth_n))
        mwin = max(2, min(FFTN, int(cfg.motion_win_s * FS)))
        smooth_k = np.ones(smooth_n) / smooth_n

        w = np.array(self.xbuf, dtype=float)[-FFTN:]
        if np.count_nonzero(~np.isnan(w)) < FFTN:
            return {"ready": False, "rssi": samp}
        w = np.nan_to_num(w, nan=float(np.nanmean(w)))
        hp = w - w.mean()                                    # high-pass (remove DC level)
        bp = np.convolve(hp, smooth_k, mode="same")          # low-pass (quantisation)
        self.last_bp = bp
        motion = float(np.sqrt(np.mean(bp[-mwin:] ** 2)))    # RMS energy in motion band
        self.i += 1

        if self.calib_request:
            self.calib_request = False
            self.calibrating = True
            self.calib_buf = []

        if self.i <= self.warmup:
            thr = 1e9
            above = False
        elif self.calibrating:
            self.calib_buf.append(motion)
            if len(self.calib_buf) >= self.calib_target:
                arr = np.array(self.calib_buf)
                # Characterise the still baseline from the QUIETEST sustained ~3s
                # sub-window so a few restless samples don't inflate the floor.
                cwin = max(2, int(3.0 * FS))
                if arr.size > cwin:
                    j0 = min(range(arr.size - cwin),
                             key=lambda j: float(np.median(arr[j:j + cwin])))
                    sub = arr[j0:j0 + cwin]
                else:
                    sub = arr
                self.floor = float(np.median(sub))
                self.mad = float(np.median(np.abs(sub - self.floor))) * 1.4826
                self.p_hi = float(np.percentile(sub, 95))
                self.calibrating = False
            thr = 1e9
            above = False
        else:
            adj_floor = max(0.0, self.floor + cfg.floor_offset)
            adj_p_hi = max(0.0, self.p_hi + cfg.floor_offset)
            thr = max(adj_floor + cfg.k * max(self.mad, 1e-3), adj_p_hi * cfg.p_margin) + 1e-3
            above = motion > thr

        if above:
            self.on_count += 1
            self.off_count = 0
        else:
            self.off_count += 1
            self.on_count = 0
        if not self.presence and self.on_count >= max(1, int(cfg.deb_on)):
            self.presence = True
        elif self.presence and self.off_count >= max(1, int(cfg.deb_off)):
            self.presence = False

        thr_disp = (0.0 if (self.calibrating or self.i <= self.warmup)
                    else max(max(0.0, self.floor + cfg.floor_offset) + cfg.k * max(self.mad, 1e-3),
                             max(0.0, self.p_hi + cfg.floor_offset) * cfg.p_margin) + 1e-3)
        return {"ready": True, "rssi": samp, "motion": motion,
                "floor": max(0.0, self.floor + cfg.floor_offset), "mad": self.mad,
                "thr": thr_disp, "above": above, "presence": self.presence,
                "calibrating": self.calibrating, "bp": bp}


class Router:
    """One real capture source: its own monitor interface + AP BSSID, its own
    frame buffer, and its own motion pipeline. Routers are interchangeable —
    Router 1 has no special data path the others lack."""

    def __init__(self, rid: int, name: str, iface: str, bssid: str,
                 enabled: bool = True) -> None:
        self.id = rid
        self.name = name
        self.iface = iface
        self.bssid = (bssid or "").lower()
        self.enabled = enabled
        self.frames: deque[tuple[float, float]] = deque(maxlen=8000)
        self.dsp = RouterDSP()
        self.out = {"id": rid, "name": name, "motion": 0.0, "rssi": None,
                    "floor": 0.0, "thr": 0.0, "presence": False, "energy": 0.0,
                    "dist": None, "enabled": enabled, "bssid": self.bssid or None,
                    "calibrating": True}


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        # Up to three real routers. Router 0 is the primary (drives the heatmap +
        # detection log + the top-level stat tiles); all run the same pipeline.
        # `frames` aliases the primary router's buffer so the single-signal fused/
        # replay path keeps writing to state.frames unchanged.
        self.routers: list[Router] = [Router(0, "Router 1", "", "")]
        self.frames: deque[tuple[float, float]] = self.routers[0].frames
        self.source = "real(rssi)"
        self.label = "unlabeled"            # ground-truth tag set from the UI
        self.calib_request = False          # set by /calibrate to (re)lock the floor
        self.logging_enabled = False        # toggled by the UI /log button
        self.components = {"csi": None, "bfi": None}
        # Provenance shown in the dashboard top bar (iface/BSSID/rate/log path).
        self.meta: dict = {}
        self.latest = {"signal": None, "rssi": None, "motion": 0.0, "floor": 0.0, "thr": 0.0,
                       "col": [0.0] * (Cfg.fftn // 2 + 1), "presence": False,
                       "fps": 0.0, "source": "real(rssi)", "label": "unlabeled",
                       "calibrating": True, "components": {"csi": None, "bfi": None},
                       "settings": {}, "freq_hz": [], "logging": False, "meta": {},
                       "routers": []}


state = State()


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def capture_real(router: "Router") -> None:
    """Per-frame RX RSSI of frames transmitted BY this router's AP, into the
    router's own buffer. Every router captures this same way — a monitor
    interface with a BPF pinned to its AP BSSID.

    A BPF of `wlan addr2 <ap>` keeps only AP->us frames so the RSSI measures one
    stable path (AP -> node) rather than a mix of our TX / control / other STAs.
    """
    bpf = [f"wlan addr2 {router.bssid}"] if router.bssid else []
    cmd = ["sudo", "-n", "tcpdump", "-i", router.iface, "-e", "-nn", "-l", "-Q", "in", *bpf]
    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            m = SIG_RE.search(line)
            if not m:
                continue
            try:
                rssi = float(m.group(1))
            except ValueError:
                continue
            with state.lock:
                router.frames.append((time.monotonic(), rssi))
        time.sleep(1.0)


def _load_npz_series(path: Path, modality: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one contract npz and return ``(x, dt)`` with shape validation."""
    with np.load(path, allow_pickle=False) as z:
        x = np.asarray(z["x"], dtype=np.float32)
        dt = np.asarray(z["dt"], dtype=np.float32)
        got = str(np.asarray(z["modality"]).item()) if "modality" in z else modality
    expected = contract.FEATURE_DIMS[modality]
    if got != modality:
        raise ValueError(f"{path} modality={got!r}, expected {modality!r}")
    if x.ndim != 2 or x.shape[1] != expected:
        raise ValueError(f"{path} x must be [T,{expected}], got {x.shape}")
    if dt.shape != (x.shape[0],):
        raise ValueError(f"{path} dt must be [{x.shape[0]}], got {dt.shape}")
    if x.shape[0] < 2:
        raise ValueError(f"{path} needs at least 2 timesteps")
    return x, dt


def _timestamps_from_dt(dt: np.ndarray, nominal_rate_hz: float) -> np.ndarray:
    """Build monotonic seconds; fall back when a capture reports zero timestamps."""
    dt = np.asarray(dt, dtype=np.float64).copy()
    if dt.size:
        dt[0] = 0.0
    usable = np.isfinite(dt) & (dt >= 0)
    positive = dt[usable & (dt > 0)]
    if positive.size == 0:
        step = 1.0 / float(nominal_rate_hz)
        return np.arange(dt.size, dtype=np.float64) * step
    fill = float(np.median(positive))
    dt[~usable] = fill
    return np.cumsum(dt)


def _feature_delta_energy(x: np.ndarray) -> np.ndarray:
    """Per-frame feature motion: robust standardise, then RMS temporal delta."""
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826
    mad = np.maximum(mad, 1e-3)
    z = np.clip((x - med) / mad, -8.0, 8.0)
    d = np.diff(z, axis=0, prepend=z[0:1])
    return np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)


def _robust_unit(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med)) * 1.4826)
    return np.clip((y - med) / max(mad, 1e-3), -6.0, 6.0)


def _resample_to_grid(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if t.size == 0 or y.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    order = np.argsort(t)
    t = np.asarray(t, dtype=np.float64)[order]
    y = np.asarray(y, dtype=np.float64)[order]
    uniq, idx = np.unique(t, return_index=True)
    y = y[idx]
    if uniq.size == 1:
        return np.full_like(grid, float(y[0]), dtype=np.float64)
    return np.interp(grid, uniq, y)


def build_fused_csi_bfi_signal(
    csi_npz: Path, bfi_npz: Path, *, fs: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Contract npz pair -> common-time fused signal plus CSI/BFI components.

    CSI and BFI keep their paper feature spaces (212 and 740). Each modality is
    reduced to temporal feature-change energy, normalized on its own scale, then
    averaged on the overlapping time grid. The dashboard heatmap therefore
    responds most strongly when both modalities move together.
    """
    csi_x, csi_dt = _load_npz_series(Path(csi_npz), "csi")
    bfi_x, bfi_dt = _load_npz_series(Path(bfi_npz), "bfi")

    csi_t = _timestamps_from_dt(csi_dt, contract.NOMINAL_RATE_HZ["csi"])
    bfi_t = _timestamps_from_dt(bfi_dt, contract.NOMINAL_RATE_HZ["bfi"])
    start = max(float(csi_t[0]), float(bfi_t[0]))
    stop = min(float(csi_t[-1]), float(bfi_t[-1]))
    if stop <= start:
        raise ValueError("CSI/BFI npz files have no overlapping time span")

    step = 1.0 / float(fs)
    grid = np.arange(start, stop + step / 2.0, step, dtype=np.float64)
    if grid.size < 2:
        raise ValueError("CSI/BFI overlap is too short for the dashboard")

    csi_e = _robust_unit(_feature_delta_energy(csi_x))
    bfi_e = _robust_unit(_feature_delta_energy(bfi_x))
    csi_g = _resample_to_grid(csi_t, csi_e, grid)
    bfi_g = _resample_to_grid(bfi_t, bfi_e, grid)
    fused = 0.5 * (csi_g + bfi_g)
    return grid - grid[0], fused.astype(np.float32), csi_g.astype(np.float32), bfi_g.astype(np.float32)


def capture_fused_npz(csi_npz: Path, bfi_npz: Path, *, replay_speed: float = 1.0) -> None:
    """Replay fused CSI+BFI motion into the live dashboard source queue."""
    grid, fused, csi, bfi = build_fused_csi_bfi_signal(csi_npz, bfi_npz, fs=cfg.fs)
    source_step = float(np.median(np.diff(grid))) if grid.size > 1 else 1.0 / cfg.fs
    sleep_s = max(0.001, source_step / max(replay_speed, 1e-6))
    i = 0
    while True:
        val = float(fused[i])
        comps = {"csi": round(float(csi[i]), 3), "bfi": round(float(bfi[i]), 3)}
        with state.lock:
            state.frames.append((time.monotonic(), val))
            state.components = comps
        i = (i + 1) % len(fused)
        time.sleep(sleep_s)




# --------------------------------------------------------------------------- #
# Logger (comprehensive, JSONL — one record per processed sample)
# --------------------------------------------------------------------------- #
class Logger:
    def __init__(self, logdir: Path) -> None:
        logdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = logdir / f"session_{ts}.jsonl"
        self.fh = self.path.open("w", buffering=1, encoding="utf-8")
        meta = {"type": "meta", "ts_utc": datetime.now(timezone.utc).isoformat(),
                "fs": cfg.fs, "smooth_n": cfg.smooth_n,
                "motion_win_s": cfg.motion_win_s,
                "k": cfg.k, "floor_offset": cfg.floor_offset,
                "deb_on": cfg.deb_on, "deb_off": cfg.deb_off,
                "p_margin": cfg.p_margin,
                "source": state.source}
        self.fh.write(json.dumps(meta) + "\n")

    def write(self, rec: dict) -> None:
        self.fh.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------------- #
# Sampler: resample -> band-pass -> motion energy -> robust threshold -> debounce
# --------------------------------------------------------------------------- #
def sampler(logger: Logger | None) -> None:
    FS = cfg.fs
    period = 1.0 / FS
    FFTN = cfg.fftn
    han = np.hanning(FFTN)
    freq_hz = [round(float(f), 2) for f in np.fft.rfftfreq(FFTN, d=1.0 / FS)]
    heat_scale = 1e-3

    while True:
        time.sleep(period)
        now = time.monotonic()
        with state.lock:
            routers = list(state.routers)
            label = state.label
            logging_enabled = state.logging_enabled
            calib_req = state.calib_request
            state.calib_request = False

        # A /calibrate press re-locks the still baseline on EVERY router at once.
        if calib_req:
            for r in routers:
                r.dsp.request_calibration()

        routers_out: list[dict] = []
        primary_res: dict | None = None
        primary_fps = 0
        for idx, r in enumerate(routers):
            if not r.enabled:
                r.out = {"id": r.id, "name": r.name, "motion": 0.0, "rssi": None,
                         "floor": 0.0, "thr": 0.0, "presence": False, "energy": 0.0,
                         "dist": None, "enabled": False, "bssid": r.bssid or None,
                         "calibrating": False}
                routers_out.append(r.out)
                continue
            with state.lock:
                recent = [rssi for (t, rssi) in r.frames if now - t < period * 1.5]
                fps = sum(1 for (t, _) in r.frames if now - t < 1.0)
            res = r.dsp.step(recent)
            if not res.get("ready"):
                r.out = {"id": r.id, "name": r.name, "motion": 0.0,
                         "rssi": res.get("rssi"), "floor": 0.0, "thr": 0.0,
                         "presence": False, "energy": 0.0, "dist": None,
                         "enabled": True, "bssid": r.bssid or None, "calibrating": True}
            else:
                r.out = {"id": r.id, "name": r.name,
                         "motion": round(res["motion"], 4),
                         "rssi": round(res["rssi"], 1),
                         "floor": round(res["floor"], 4),
                         "thr": round(res["thr"], 4),
                         "presence": bool(res["presence"]),
                         "energy": round(res["motion"], 4),
                         "dist": None, "enabled": True,
                         "bssid": r.bssid or None,
                         "calibrating": bool(res["calibrating"])}
            routers_out.append(r.out)
            if idx == 0:
                primary_res = res
                primary_fps = fps

        # The PRIMARY router (router 0) drives the heatmap, the detection log, and
        # the top-level stat tiles the existing UI reads. Every router's own
        # motion/threshold/presence is carried in `routers`.
        col = [0.0] * (FFTN // 2 + 1)
        p = primary_res if (primary_res and primary_res.get("ready")) else None
        if p is not None:
            mag = np.abs(np.fft.rfft(p["bp"] * han))        # heatmap from band-passed sig
            spec_raw = np.log1p(mag)
            col_hi = float(np.percentile(spec_raw, 95))
            if col_hi > heat_scale:
                heat_scale = col_hi
            else:
                heat_scale = max(col_hi, heat_scale * cfg.heat_decay, 1e-3)
            spec = np.clip(spec_raw / (heat_scale + 1e-9), 0.0, 1.0)
            # Keep static room texture from reading as activity in the visual.
            freq_arr = np.asarray(freq_hz, dtype=np.float64)
            spec[freq_arr < 0.3] *= 0.10
            spec[freq_arr > 3.0] *= 0.30
            spec = np.clip((spec - cfg.heat_floor) / max(1.0 - cfg.heat_floor, 1e-6), 0.0, 1.0)
            col = [round(float(c), 4) for c in spec]

        samp = p["rssi"] if p else (primary_res.get("rssi") if primary_res else None)
        motion = p["motion"] if p else 0.0
        floor = p["floor"] if p else 0.0
        thr_disp = p["thr"] if p else 0.0
        mad = p["mad"] if p else 0.0
        presence = bool(p["presence"]) if p else False
        calibrating = bool(p["calibrating"]) if p else True

        if logger and logging_enabled and p is not None:
            logger.write({"type": "s", "t": round(now, 3),
                          "rssi": round(samp, 2), "fps": primary_fps,
                          "motion": round(motion, 4), "floor": round(floor, 4),
                          "mad": round(mad, 4), "thr": round(thr_disp, 4),
                          "presence": presence, "calibrating": calibrating,
                          "label": label, "col": col,
                          "routers": [ro.get("motion") for ro in routers_out]})

        with state.lock:
            comps = dict(state.components)
            settings = current_settings()
            state.latest = {"signal": round(samp, 3) if samp is not None else None,
                            "rssi": (round(samp, 1) if (samp is not None
                                     and state.source == "real(rssi)") else None),
                            "motion": round(motion, 3),
                            "floor": round(floor, 3),
                            "thr": round(thr_disp, 3),
                            "mad": round(mad, 3),
                            "col": col, "presence": presence, "fps": primary_fps,
                            "source": state.source, "label": label,
                            "calibrating": calibrating, "components": comps,
                            "settings": settings, "freq_hz": freq_hz,
                            "heat_scale": round(heat_scale, 4),
                            "logging": logging_enabled, "meta": state.meta,
                            "routers": routers_out}


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def detect_ap_bssid(client_iface: str) -> str:
    """Return the lowercased BSSID the client iface is associated to, or "".

    Parses `iw dev <iface> link` ("Connected to <bssid> (on ...)"). Needs no
    sudo. Any subprocess error / not-associated / no match yields "".
    """
    try:
        out = subprocess.run(["iw", "dev", client_iface, "link"],
                             capture_output=True, text=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"Connected to ([0-9a-fA-F:]{17})", out)
    return m.group(1).lower() if m else ""


def ensure_monitor(client_iface: str, mon_iface: str) -> str:
    if subprocess.run(["iw", "dev", mon_iface, "info"], capture_output=True).returncode != 0:
        subprocess.run(["sudo", "-n", "iw", "dev", client_iface, "interface",
                        "add", mon_iface, "type", "monitor"], check=False)
        subprocess.run(["sudo", "-n", "ip", "link", "set", mon_iface, "up"], check=False)
    return mon_iface


def start_traffic(gateway: str, n: int = 20) -> list[subprocess.Popen]:
    return [subprocess.Popen(["ping", "-q", "-i", "0.2", gateway],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(max(1, n))]


# --------------------------------------------------------------------------- #
# HTTP: SSE stream, ground-truth labeling, dashboard page
# --------------------------------------------------------------------------- #
def current_settings() -> dict:
    return {
        "k": round(float(cfg.k), 3),
        "floor_offset": round(float(cfg.floor_offset), 4),
        "smooth_n": int(cfg.smooth_n),
        "motion_win_s": round(float(cfg.motion_win_s), 3),
        "deb_on": int(cfg.deb_on),
        "deb_off": int(cfg.deb_off),
        "p_margin": round(float(cfg.p_margin), 3),
    }


def _set_float(qs: dict, key: str, lo: float, hi: float) -> float | None:
    try:
        return max(lo, min(hi, float(qs[key][0])))
    except (KeyError, ValueError, TypeError):
        return None


def _set_int(qs: dict, key: str, lo: int, hi: int) -> int | None:
    try:
        return max(lo, min(hi, int(float(qs[key][0]))))
    except (KeyError, ValueError, TypeError):
        return None


STATIC_DIR = Path(__file__).resolve().parent / "static"


def _static_response(path: Path, content_type: str) -> tuple[int, bytes, str]:
    try:
        body = path.read_bytes()
    except OSError:
        return 404, b"not found", "text/plain; charset=utf-8"
    return 200, body, content_type


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/label":
            st = (parse_qs(parsed.query).get("state", ["unlabeled"])[0])[:24]
            with state.lock:
                state.label = st
            self.send_response(200); self.send_header("Content-Length", "2")
            self.end_headers(); self.wfile.write(b"ok")
            return
        if parsed.path == "/calibrate":
            with state.lock:
                state.calib_request = True
            self.send_response(200); self.send_header("Content-Length", "2")
            self.end_headers(); self.wfile.write(b"ok")
            return
        if parsed.path == "/log":
            qs = parse_qs(parsed.query)
            on = qs.get("on", ["1"])[0].lower() not in ("0", "false", "off")
            with state.lock:
                state.logging_enabled = on
            body = json.dumps({"logging": on}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
            return
        if parsed.path == "/set":
            qs = parse_qs(parsed.query)
            val = _set_float(qs, "k", 0.5, 20.0)
            if val is not None:
                cfg.k = val
            val = _set_float(qs, "floor_offset", -1.0, 2.0)
            if val is not None:
                cfg.floor_offset = val
            val_i = _set_int(qs, "smooth_n", 1, 32)
            if val_i is not None:
                cfg.smooth_n = val_i
            val = _set_float(qs, "motion_win_s", 0.1, 5.0)
            if val is not None:
                cfg.motion_win_s = val
            val_i = _set_int(qs, "deb_on", 1, 80)
            if val_i is not None:
                cfg.deb_on = val_i
            val_i = _set_int(qs, "deb_off", 1, 120)
            if val_i is not None:
                cfg.deb_off = val_i
            val = _set_float(qs, "p_margin", 1.0, 4.0)
            if val is not None:
                cfg.p_margin = val
            body = json.dumps(current_settings()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)
            return
        if parsed.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with state.lock:
                        payload = json.dumps(state.latest)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    # Emit one column per sampler tick so a heatmap/trace column
                    # equals one sample (true time). The JS COL_DT must track this.
                    time.sleep(1.0 / cfg.fs)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
        if parsed.path in ("/", "/live_motion.html"):
            code, body, content_type = _static_response(
                STATIC_DIR / "live_motion.html", "text/html; charset=utf-8"
            )
        elif parsed.path == "/static/live_motion.css":
            code, body, content_type = _static_response(
                STATIC_DIR / "live_motion.css", "text/css; charset=utf-8"
            )
        elif parsed.path == "/static/live_motion.js":
            code, body, content_type = _static_response(
                STATIC_DIR / "live_motion.js", "application/javascript; charset=utf-8"
            )
        else:
            code, body, content_type = 404, b"not found", "text/plain; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m viz.live_motion")
    ap.add_argument("--client-iface", default="wlp1s0")
    ap.add_argument("--mon-iface", default="mon1")
    ap.add_argument("--ap-bssid", default="",
                    help="only use frames from this AP (consistent RSSI source); "
                         "if omitted, the BSSID is auto-detected from the client "
                         "interface's current association; pass an explicit value "
                         "to override; empty + no association = all frames")
    ap.add_argument("--router", action="append", default=[], metavar="IFACE[:BSSID]",
                    help="a real router source: a monitor interface, optionally "
                         "pinned to an AP BSSID (e.g. mon1:AA:BB:CC:DD:EE:F4). Repeat "
                         "up to 3 times for Routers 1-3 — each is captured identically "
                         "and can watch a different AP. If omitted, one router is built "
                         "from --mon-iface/--client-iface as today.")
    ap.add_argument("--gateway", default="198.51.100.1")
    ap.add_argument("--pings", type=int, default=20)
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--csi-npz", default="",
                    help="processed CSI .npz to fuse with --bfi-npz")
    ap.add_argument("--bfi-npz", default="",
                    help="processed BFI .npz to fuse with --csi-npz")
    ap.add_argument("--replay-speed", type=float, default=1.0,
                    help="speed multiplier for --csi-npz/--bfi-npz replay")
    ap.add_argument("--no-traffic", action="store_true")
    ap.add_argument("--logdir", default="data/reports/motion_logs")
    ap.add_argument("--no-log", action="store_true")
    # DSP knobs (iterate on these)
    ap.add_argument("--k", type=float, default=cfg.k)
    ap.add_argument("--floor-offset", type=float, default=cfg.floor_offset)
    ap.add_argument("--p-margin", type=float, default=cfg.p_margin)
    ap.add_argument("--smooth-n", type=int, default=cfg.smooth_n)
    ap.add_argument("--motion-win-s", type=float, default=cfg.motion_win_s)
    ap.add_argument("--deb-on", type=int, default=cfg.deb_on)
    ap.add_argument("--deb-off", type=int, default=cfg.deb_off)
    args = ap.parse_args(argv)

    cfg.k = args.k; cfg.floor_offset = args.floor_offset; cfg.p_margin = args.p_margin
    cfg.smooth_n = args.smooth_n
    cfg.motion_win_s = args.motion_win_s
    cfg.deb_on = args.deb_on; cfg.deb_off = args.deb_off

    procs: list[subprocess.Popen] = []
    if bool(args.csi_npz) != bool(args.bfi_npz):
        raise SystemExit("--csi-npz and --bfi-npz must be supplied together")

    if args.csi_npz and args.bfi_npz:
        state.source = "fused(csi+bfi)"
        # Single fused signal → one primary router; state.frames aliases its buffer.
        state.routers = [Router(0, "Router 1", "", "")]
        state.frames = state.routers[0].frames
        threading.Thread(
            target=capture_fused_npz,
            args=(Path(args.csi_npz), Path(args.bfi_npz)),
            kwargs={"replay_speed": args.replay_speed},
            daemon=True,
        ).start()
    elif args.router:
        # Explicit multi-router mode: each --router is a ready monitor interface,
        # optionally pinned to its own AP BSSID. Up to three, captured alike — no
        # router is derived from another.
        state.source = "real(rssi)"
        routers = []
        for i, spec in enumerate(args.router[:3]):
            iface, sep, bssid = spec.partition(":")
            iface = iface.strip()
            bssid = bssid.strip() if sep else ""
            if not bssid and i == 0 and not args.ap_bssid:
                bssid = detect_ap_bssid(args.client_iface)
            routers.append(Router(i, f"Router {i + 1}", iface, bssid))
            print(f"[viz] Router {i + 1}: iface={iface} "
                  f"bssid={bssid or '(all frames)'}", flush=True)
        state.routers = routers
        state.frames = state.routers[0].frames
        if not args.no_traffic:
            procs = start_traffic(args.gateway, args.pings)
        for r in state.routers:
            threading.Thread(target=capture_real, args=(r,), daemon=True).start()
        ap_bssid = state.routers[0].bssid
    else:
        state.source = "real(rssi)"
        # Legacy single-router path (unchanged behaviour): build Router 1 from
        # --mon-iface / --client-iface with the associated (or explicit) BSSID.
        # The association lives on the client iface, NOT the monitor iface.
        if args.ap_bssid:
            ap_bssid = args.ap_bssid.lower()
            print(f"[viz] ap-bssid={ap_bssid} (explicit)", flush=True)
        else:
            ap_bssid = detect_ap_bssid(args.client_iface)
            if ap_bssid:
                print(f"[viz] ap-bssid={ap_bssid} (auto-detected from "
                      f"{args.client_iface})", flush=True)
            else:
                print(f"[viz] WARNING: no AP association on {args.client_iface} "
                      f"and no --ap-bssid; capturing ALL frames", flush=True)
        mon = ensure_monitor(args.client_iface, args.mon_iface)
        state.routers = [Router(0, "Router 1", mon, ap_bssid)]
        state.frames = state.routers[0].frames
        if not args.no_traffic:
            procs = start_traffic(args.gateway, args.pings)
        threading.Thread(target=capture_real, args=(state.routers[0],),
                         daemon=True).start()

    logger = None if args.no_log else Logger(Path(args.logdir))
    state.meta = {
        "client_iface": args.client_iface,
        "mon_iface": args.mon_iface,
        "ap_bssid": (locals().get("ap_bssid") or "—"),
        "fs": cfg.fs,
        "port": args.port,
        "log_path": "off" if args.no_log else str(logger.path),
    }
    state.latest["meta"] = state.meta
    state.latest["source"] = state.source
    threading.Thread(target=sampler, args=(logger,), daemon=True).start()

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[viz] live motion on http://0.0.0.0:{args.port}/  source={state.source}  "
          f"k={cfg.k} smooth={cfg.smooth_n} log={'off' if args.no_log else logger.path}",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
