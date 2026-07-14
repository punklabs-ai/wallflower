"""Traffic generator node agent (drives WiFi sounding for BFI + CSI).

Per BFId (Todt et al., CCS '25 sec. 5.1) two flows run during a trial:
  * ~200 Mb/s TCP downlink from a lab server through AP-BFI to the Wi-Fi client:
    high throughput makes the AP beamform frequently -> frequent Compressed
    Beamforming Reports (the BFI signal).
  * ~30 Kb/s UDP for AP-CSI: a light, steady flow keeps packets arriving on the
    CSI monitor so CSI is continuously observable.

These are ordinary iperf3 flows between the configured hosts.

  detect  -> report iperf3 availability + configured rates
  start   -> launch the TCP (BFI, reverse/downlink) and/or UDP (CSI) iperf3
             flows against the lab servers; pidfiles let stop terminate them
  stop    -> kill the traffic processes
  status  -> running flows + pids
  health  -> tool availability / kernel summary

iperf3 is ABSENT on this pilot node. The agent does NOT crash: it prints an
install hint and notes a stdlib socket fallback (a minimal TCP/UDP blaster) that
an operator can enable. We do not silently start the fallback because saturating
a link is something an operator should opt into explicitly (--allow-fallback).

STDLIB-ONLY (no third-party imports). iperf3 is an external binary, not a Python
dependency.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from wallflower.contract import AP_BFI_CHANNEL, AP_CSI_CHANNEL

from . import common

AGENT = "traffic_agent"

# Defaults mirror configs/lab.yaml -> traffic (paper sec 5.1). Read live so a
# lab can retune without code changes.
DEFAULT_BFI_TCP_MBPS = 200
DEFAULT_CSI_UDP_KBPS = 30
IPERF3_PORT_BFI = 5201
IPERF3_PORT_CSI = 5202


def _traffic_cfg() -> dict:
    lab = common.load_yaml(common.LAB_YAML)
    t = lab.get("traffic") if isinstance(lab.get("traffic"), dict) else {}
    t = t or {}
    return {
        "bfi_tcp_mbps": t.get("bfi_tcp_mbps", DEFAULT_BFI_TCP_MBPS),
        "csi_udp_kbps": t.get("csi_udp_kbps", DEFAULT_CSI_UDP_KBPS),
    }


def _install_hint() -> str:
    return common.privileged_hint(["apt-get", "install", "-y", "iperf3"])


def detect(node: str | None) -> dict:
    cfg = _traffic_cfg()
    have = common.have_tool("iperf3")
    return common.emit(
        AGENT, "detect", True, node=node or common.hostname(),
        iperf3_available=have,
        iperf3_path=common.tool_path("iperf3"),
        traffic=cfg,
        ap_bfi_channel=AP_BFI_CHANNEL, ap_csi_channel=AP_CSI_CHANNEL,
        install_hint=None if have else _install_hint(),
        fallback_note=None if have else
            "stdlib socket fallback available with --allow-fallback (minimal "
            "TCP/UDP blaster; opt-in because it saturates the link).",
    )


def _launch(cmd: list[str], log_path: Path):
    logf = open(log_path, "ab")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)


def start(args) -> dict:
    out_dir = common.default_out_dir(args)
    if out_dir is None:
        return common.emit(AGENT, "start", False, error="cannot resolve out-dir")
    cfg = _traffic_cfg()
    node = args.node or common.hostname()

    bfi_server = getattr(args, "bfi_server", None)
    csi_server = getattr(args, "csi_server", None)

    if not common.have_tool("iperf3"):
        # Degrade gracefully: do not crash, do not auto-saturate the link.
        if not getattr(args, "allow_fallback", False):
            return common.emit(
                AGENT, "start", False, node=node,
                error="iperf3 not installed; traffic not started.",
                install_hint=_install_hint(),
                fallback_note="re-run with --allow-fallback to use the stdlib "
                              "socket blaster instead, or install iperf3.",
                traffic=cfg,
            )
        # Stdlib fallback is intentionally a documented stub: a real blaster
        # needs lab server endpoints. We record intent and the exact endpoints
        # required rather than blindly flooding.
        return common.emit(
            AGENT, "start", False, node=node,
            error="stdlib fallback requires --bfi-server/--csi-server "
                  "endpoints; none provided.",
            traffic=cfg,
            fallback_note="provide --bfi-server HOST and --csi-server HOST "
                          "to enable the socket fallback.",
        )

    logs = common.logs_dir_for(out_dir)
    logs.mkdir(parents=True, exist_ok=True)
    started: list[dict] = []
    pids: list[int] = []

    # BFI: high-rate TCP downlink from the lab server, through the AP, to this
    # associated Wi-Fi client. Reverse mode is the important bit: AP->STA
    # downlink traffic is what should trigger explicit sounding and CBRs.
    if bfi_server:
        mbps = cfg["bfi_tcp_mbps"]
        cmd = ["iperf3", "-c", bfi_server, "-p", str(IPERF3_PORT_BFI),
               "-R", "-P", "4", "-b", f"{mbps}M", "-t", "0", "-Z"]
        proc = _launch(cmd, logs / f"{AGENT}_bfi.log")
        pids.append(proc.pid)
        started.append({"flow": "bfi_tcp", "server": bfi_server,
                        "direction": "downlink_reverse", "parallel": 4,
                        "mbps": mbps, "pid": proc.pid})

    # CSI: light UDP toward AP-CSI server.
    if csi_server:
        kbps = cfg["csi_udp_kbps"]
        cmd = ["iperf3", "-c", csi_server, "-p", str(IPERF3_PORT_CSI),
               "-u", "-b", f"{kbps}K", "-t", "0"]
        proc = _launch(cmd, logs / f"{AGENT}_csi.log")
        pids.append(proc.pid)
        started.append({"flow": "csi_udp", "server": csi_server,
                        "kbps": kbps, "pid": proc.pid})

    if not started:
        return common.emit(
            AGENT, "start", False, node=node, traffic=cfg,
            error="no traffic servers specified; pass --bfi-server and/or "
                  "--csi-server (iperf3 servers).",
        )

    common.write_pidfile(out_dir, AGENT, pids[0],
                        meta={"flows": started, "all_pids": pids})
    return common.emit(AGENT, "start", True, node=node, flows=started,
                      pids=pids, traffic=cfg)


def stop(args) -> dict:
    out_dir = common.default_out_dir(args)
    if out_dir is None:
        return common.emit(AGENT, "stop", False, error="cannot resolve out-dir")
    info = common.read_pidfile(out_dir, AGENT)
    if not info:
        # Idempotent: nothing was started (never started or already stopped)
        # -> a clean stop, not a failure.
        return common.emit(AGENT, "stop", True, mode="no_pidfile",
                           note="no traffic flow to stop")
    killed: list[int] = []
    failed: list[dict] = []
    import os
    import signal
    for pid in info.get("all_pids", [info.get("pid")]):
        if pid is None:
            continue
        if not common.pid_alive(int(pid)):
            continue
        try:
            os.kill(int(pid), signal.SIGTERM)
            killed.append(int(pid))
        except PermissionError:
            failed.append({"pid": pid,
                           "hint": common.privileged_hint(["kill", "-TERM", str(pid)])})
        except Exception as exc:
            failed.append({"pid": pid, "error": str(exc)})
    common.pidfile_path(out_dir, AGENT).unlink(missing_ok=True)
    return common.emit(AGENT, "stop", not failed, killed=killed, failed=failed)


def status(args) -> dict:
    out_dir = common.default_out_dir(args)
    info = common.read_pidfile(out_dir, AGENT) if out_dir else None
    flows = (info or {}).get("flows", [])
    running = [f for f in flows
               if f.get("pid") and common.pid_alive(int(f["pid"]))]
    return common.emit(AGENT, "status", True, node=args.node or common.hostname(),
                      running_flows=running, configured=flows,
                      iperf3_available=common.have_tool("iperf3"))


def health(args) -> dict:
    return common.emit(AGENT, "health", True, node=args.node or common.hostname(),
                      iperf3_available=common.have_tool("iperf3"),
                      install_hint=None if common.have_tool("iperf3") else _install_hint(),
                      traffic=_traffic_cfg(), kernel=common.kernel_info(),
                      tools=common.tool_availability())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m nodes.traffic_agent")
    common.add_common_args(parser)
    parser.add_argument("--bfi-server", dest="bfi_server",
                        help="iperf3 server reachable via AP-BFI")
    parser.add_argument("--csi-server", dest="csi_server",
                        help="iperf3 server reachable via AP-CSI")
    parser.add_argument("--allow-fallback", dest="allow_fallback",
                        action="store_true",
                        help="opt in to the stdlib socket blaster when iperf3 "
                             "is missing (saturates the link)")
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
