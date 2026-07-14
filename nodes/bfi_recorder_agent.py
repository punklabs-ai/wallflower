"""Passive BFI recorder node agent (monitor capture of beamforming reports).

The recorder is a monitor-mode radio on the AP-BFI channel (operative pilot:
5 GHz ch36 / 80 MHz) that passively captures 802.11 management/action frames carrying Compressed
Beamforming Reports (CBR) into bfi_recorder.pcapng. Per BFId (Todt et al., CCS
'25), these CBRs are the raw material for the BFI identity-inference features.

A specific target network (BSSID, and ideally SSID) must be configured via
--bssid (or BFID_AP_BFI_BSSID env, or an `ap_bfi` block in configs/lab.yaml).
The tcpdump filter is pinned to that BSSID; the agent will not start a wildcard
"capture everything" sniff.

  detect  -> identify the recorder radio + the configured target network
  start   -> monitor on AP-BFI (root needed; prints exact `iw` cmds otherwise),
             launch capture.bfi_pcap pinned to the target BSSID -> .pcapng
  stop    -> kill the capture process from the pidfile
  status  -> capture-running + interface state
  health  -> driver/kernel/clock/tool summary

STDLIB-ONLY. Monitor mode + raw capture need root; missing privilege is
surfaced as exact commands, never a crash.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from wallflower import contract
from wallflower.contract import (
    AP_BFI_CHANNEL,
    BAND_GHZ,
    FIVEGHZ_WIDTH_MHZ,
    WIDTH_MHZ,
    bfi_recorder_name,
)

from . import common
from capture import bfi_pcap

AGENT = "bfi_recorder_agent"
ROLE = "bfi_recorder"

_BSSID_RE = re.compile(r"^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$")


def _resolve_recorder_radio(node: str | None) -> dict:
    """Recorder radio: explicit 'bfi_recorder' role if present, else 'bfi'."""
    r = common.resolve_role_iface(ROLE, node)
    if r["iface"] and r["source"] != "none":
        return r
    # Fall back to the bfi-role radio (pilot collapses recorder onto node1).
    return common.resolve_role_iface("bfi", node)


def _target(args) -> dict:
    bssid = getattr(args, "bssid", None) or os.environ.get("BFID_AP_BFI_BSSID")
    ssid = getattr(args, "ssid", None) or os.environ.get("BFID_AP_BFI_SSID")
    if not bssid or not ssid:
        lab = common.load_yaml(common.LAB_YAML)
        ap = lab.get("ap_bfi") if isinstance(lab.get("ap_bfi"), dict) else {}
        ap = ap or {}
        bssid = bssid or ap.get("bssid")
        ssid = ssid or ap.get("ssid")
    return {"bssid": bssid, "ssid": ssid}


def _capture_rf(args, radio: dict) -> dict:
    """Resolve the monitor channel/band/width for the BFI capture.

    Precedence: CLI --channel/--band > the ``ap_bfi`` block in
    configs/lab.yaml > the radio-role channel / contract constants. The band is
    load-bearing: a bare channel number is ambiguous across bands (ch 36 exists
    on both 5 and 6 GHz), and tuning the wrong band fails (6 GHz is regulatory-
    locked; there is no AP on the nominal ch 85).
    """
    lab = common.load_yaml(common.LAB_YAML)
    ap = lab.get("ap_bfi") if isinstance(lab.get("ap_bfi"), dict) else {}
    ap = ap or {}
    band = getattr(args, "band", None) or ap.get("band_ghz") or BAND_GHZ
    band = int(band)
    channel = (getattr(args, "channel", None)
               or ap.get("channel") or radio.get("channel") or AP_BFI_CHANNEL)
    channel = int(channel)
    # Width follows the resolved band so it always equals what monitor_freq_args
    # actually tunes (80 MHz on 5 GHz, 160 MHz on 6 GHz) — never the stale
    # ap_bfi.width_mhz when a --band override disagrees with the config band.
    width = FIVEGHZ_WIDTH_MHZ if band == 5 else WIDTH_MHZ
    return {"channel": channel, "band_ghz": band, "width_mhz": int(width)}


def _monitor_cmds(iface: str, channel: int, band_ghz: int = BAND_GHZ) -> list[list[str]]:
    # Both bands tune by FREQUENCY (6 GHz channel numbers collide with legacy
    # numbering; 5 GHz we set an explicit centre so the 80 MHz block is pinned).
    return [
        ["ip", "link", "set", iface, "down"],
        ["iw", "dev", iface, "set", "type", "monitor"],
        ["ip", "link", "set", iface, "up"],
        ["iw", "dev", iface, "set", "freq",
         *contract.monitor_freq_args(channel, band_ghz)],
    ]


def detect(node: str | None) -> dict:
    radio = _resolve_recorder_radio(node)
    radios = common.detect_radios()
    rf = _capture_rf(argparse.Namespace(channel=None, band=None), radio)
    ok = bool(radio["iface"])
    return common.emit(AGENT, "detect", ok, node=radio["node"], role=ROLE,
                      recorder_radio=radio, radios=radios,
                      ap_channel=rf["channel"], band_ghz=rf["band_ghz"],
                      width_mhz=rf["width_mhz"])


def _data_root() -> str:
    try:
        return common.load_yaml(common.LAB_YAML).get("data_root", "data")
    except Exception:
        return "data"


def _delegate(fn, **kw) -> dict:
    """Run a capture-wrapper fn in-process, suppressing its own stdout JSON so the
    node agent emits exactly one structured object (NODE-AGENT contract)."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return fn(**kw)


def start(args) -> dict:
    radio = _resolve_recorder_radio(args.node)
    iface = radio["iface"]
    if not (args.participant and args.style and args.trial):
        return common.emit(AGENT, "start", False, node=radio["node"],
                          error="participant/style/trial required to resolve trial dir")

    target = _target(args)
    # Require an explicit target AP BSSID (no wildcard capture).
    if not target["bssid"]:
        return common.emit(
            AGENT, "start", False, node=radio["node"], iface=iface,
            error="refusing capture: an explicit target AP BSSID is required; "
                  "wildcard capture is not supported. "
                  "Provide --bssid (and --ssid) or set BFID_AP_BFI_BSSID or an "
                  "ap_bfi block in configs/lab.yaml.",
            target=target,
        )
    if not _BSSID_RE.match(str(target["bssid"])) or target["bssid"].lower() in (
        "ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00"
    ):
        return common.emit(
            AGENT, "start", False, node=radio["node"], iface=iface,
            error=f"refusing capture: invalid/wildcard target BSSID {target['bssid']!r}; "
                  "a single concrete AP-BFI BSSID is required.",
            target=target,
        )

    rf = _capture_rf(args, radio)
    channel, band_ghz, width_mhz = rf["channel"], rf["band_ghz"], rf["width_mhz"]

    # --- host backend: needs a real monitor-capable host iface ----------------
    if not iface:
        return common.emit(AGENT, "start", False, node=radio["node"], role=ROLE,
                          error="no recorder-role AX210 radio resolved")

    # --- privileged monitor setup for a REAL capture --------------------------
    monitor_note = "not_attempted"
    if iface and common._iw_dev_types().get(iface) != "monitor":
        mon_cmds = _monitor_cmds(iface, channel, band_ghz)
        if all(common.can_run_priv(c) for c in mon_cmds):
            # root, or scoped passwordless sudo for iw/ip -> monitor on AP-BFI ch.
            for cmd in mon_cmds:
                rc, _, err = common.run_priv(cmd)
                if rc != 0:
                    return common.emit(AGENT, "start", False, node=radio["node"],
                                      iface=iface,
                                      error=f"monitor setup failed: "
                                            f"{' '.join(common.priv_cmd(cmd))}: {err.strip()}",
                                      privileged_commands=[common.privileged_hint(c)
                                                           for c in mon_cmds])
            monitor_note = ("by_root" if common.is_root() else "by_sudo") \
                if common._iw_dev_types().get(iface) == "monitor" else "incomplete"
        else:
            monitor_note = "needs_privilege"

    # Delegate to capture.bfi_pcap, which pins the tcpdump filter to the target
    # BSSID and returns the operator commands when tcpdump / root is unavailable.
    res = _delegate(
        bfi_pcap.start, participant=args.participant, style=args.style,
        trial=args.trial, bssid=target["bssid"], data_root=_data_root(),
        mon_iface=(iface or "mon_bfi"), phy=radio.get("phy", "<phy>"),
    )
    return common.emit(
        AGENT, "start", bool(res.get("ok")), node=radio["node"], iface=iface,
        role=ROLE, backend="host", channel=channel, band_ghz=band_ghz,
        width_mhz=width_mhz, target=target,
        monitor_setup=monitor_note, mode=res.get("mode"), out=res.get("out_pcap"),
        capture_scope="single-bssid", capture=res,
    )


def stop(args) -> dict:
    if not (args.participant and args.style and args.trial):
        return common.emit(AGENT, "stop", False,
                           error="participant/style/trial required")
    data_root = _data_root()
    res = _delegate(bfi_pcap.stop, participant=args.participant, style=args.style,
                    trial=args.trial, data_root=data_root)
    return common.emit(AGENT, "stop", bool(res.get("ok")),
                       backend="host", mode=res.get("mode"), capture=res)


def status(args) -> dict:
    radio = _resolve_recorder_radio(args.node)
    out_dir = common.default_out_dir(args)
    info = common.read_pidfile(out_dir, AGENT) if out_dir else None
    running = bool(info and common.pid_alive(int(info.get("pid", -1))))
    return common.emit(AGENT, "status", True, node=radio["node"],
                      iface=radio["iface"], backend="host", running=running,
                      pid=(info or {}).get("pid"),
                      iface_type=common._iw_dev_types().get(radio["iface"], "unknown"),
                      out=(info or {}).get("out"))


def health(args) -> dict:
    radio = _resolve_recorder_radio(args.node)
    return common.emit(AGENT, "health", True, node=radio["node"], role=ROLE,
                      recorder_radio=radio, kernel=common.kernel_info(),
                      clock=common.clock_sources(), tools=common.tool_availability(),
                      radios=common.detect_radios())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m nodes.bfi_recorder_agent")
    common.add_common_args(parser)
    parser.add_argument("--bssid", help="target AP-BFI BSSID to record "
                                       "(required; no wildcard capture)")
    parser.add_argument("--ssid", help="AP-BFI SSID (optional label)")
    parser.add_argument("--channel", type=int,
                        help="monitor channel to tune (overrides ap_bfi config / "
                             "radio role; e.g. 36 for the operative 5 GHz AP)")
    parser.add_argument("--band", type=int, choices=(5, 6),
                        help="band in GHz for the monitor tune (overrides ap_bfi "
                             "config; ch numbers collide across bands so this "
                             "disambiguates — 5 for the operative ASUS ch 36)")
    args = parser.parse_args(argv)
    handler = {
        "detect": lambda: detect(args.node),
        "start": lambda: start(args),
        "stop": lambda: stop(args),
        "status": lambda: status(args),
        "health": lambda: health(args),
    }[args.action]
    result = handler()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
