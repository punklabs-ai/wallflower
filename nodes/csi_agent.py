"""CSI capture node agent (radio A, monitor mode on AP-CSI channel).

Per BFId (Todt et al., CCS '25) sec. 5.1, CSI is captured from a monitor-mode
radio tuned to the AP-CSI channel (6 GHz ch 37, 160 MHz) while a light UDP flow
(~30 Kb/s) keeps packets flowing. This agent owns the *node-side* lifecycle:

  detect  -> list AX210 radios + identify the csi-role radio
  start   -> put the csi radio into monitor on ch 37 (needs root: prints the
             exact `iw` commands if not root) and launch the capture wrapper
             (capture.csi_picoscenes) writing csi_p<perspective>.raw
  stop    -> kill the capture process recorded in the pidfile
  status  -> report whether capture is running + interface state
  health  -> driver/kernel/firmware/clock/tool summary

STDLIB-ONLY. Capture is pinned to the configured AP-CSI channel. The actual
decode/parse of the .raw is done off-node by parsers/.

Privilege model: sudo on a node requires a password (no non-interactive root).
Anything privileged is surfaced as an exact command for the operator to run; the
agent never attempts to escalate and never crashes for lack of root.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from wallflower import contract
from wallflower.contract import (
    AP_CSI_CHANNEL,
    BAND_GHZ,
    WIDTH_MHZ,
    csi_raw_name,
)

from . import common

AGENT = "csi_agent"
ROLE = "csi"


def _monitor_cmds(iface: str, channel: int) -> list[list[str]]:
    """The privileged sequence to bring `iface` into monitor on the channel.

    6 GHz must be tuned by FREQUENCY, not channel number (`iw set channel 85`
    fails as "Unknown channel" because 85 collides with legacy numbering). We use
    `iw ... set freq <control> 160MHz <center>` from contract.monitor_freq_args.
    """
    return [
        ["ip", "link", "set", iface, "down"],
        ["iw", "dev", iface, "set", "type", "monitor"],
        ["ip", "link", "set", iface, "up"],
        ["iw", "dev", iface, "set", "freq", *contract.monitor_freq_args(channel)],
    ]


def _iface_is_monitor(iface: str) -> bool:
    return common._iw_dev_types().get(iface) == "monitor"


def detect(node: str | None) -> dict:
    from capture import csi_picoscenes  # backend tool detection (FeitCSI/PicoScenes)

    radios = common.detect_radios()
    csi = common.resolve_role_iface(ROLE, node)
    tool = csi_picoscenes.available()
    ok = bool(csi["iface"]) and any(r["is_ax210"] for r in radios)
    return common.emit(
        AGENT, "detect", ok,
        node=csi["node"],
        role=ROLE,
        csi_radio=csi,
        radios=radios,
        ap_channel=AP_CSI_CHANNEL,
        band_ghz=BAND_GHZ,
        width_mhz=WIDTH_MHZ,
        csi_backend=tool.get("csi_backend"),
        csi_tool_available=tool.get("csi_tool_available"),
        feitcsi_available=tool.get("feitcsi_available"),
        picoscenes_available=tool.get("picoscenes_available"),
    )


def _data_root() -> str:
    try:
        return common.load_yaml(common.LAB_YAML).get("data_root", "data")
    except Exception:
        return "data"


def _delegate(fn, **kw) -> dict:
    """Run a capture-wrapper fn in-process while suppressing its own stdout JSON,
    so the node agent emits exactly one structured object (NODE-AGENT contract).
    The wrapper self-detaches real captures via start_new_session, so calling it
    in-process is safe for the long-running capture path."""
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return fn(**kw)


def start(args) -> dict:
    from capture import csi_picoscenes  # delegated capture wrapper

    csi = common.resolve_role_iface(ROLE, args.node)
    iface = csi["iface"]
    perspective = args.perspective or csi.get("perspective") or 1
    if not (args.participant and args.style and args.trial):
        return common.emit(
            AGENT, "start", False, node=csi["node"],
            error="participant/style/trial required to resolve the trial dir",
        )
    if not iface:
        return common.emit(
            AGENT, "start", False, node=csi["node"], role=ROLE,
            error="no csi-role AX210 radio resolved",
        )
    channel = csi.get("channel") or AP_CSI_CHANNEL

    # Which real CSI backend (if any) is installed dictates whether a monitor VIF
    # flip is even needed: FeitCSI tunes the card itself (no monitor dance);
    # PicoScenes reads from a monitor interface we'd set up via iw+ip.
    backend, _tool = csi_picoscenes.csi_tool()

    # --- best-effort privileged monitor setup for a REAL capture --------------
    # Skipped for the FeitCSI backend, which tunes the card to ch37 directly
    # (flipping the radio to monitor here would be wrong AND could disturb the
    # live stack). When not root, we do NOT block: the wrapper still emits the
    # exact operator_commands (and returns ok=False if no tool/privilege exists).
    needs_monitor_vif = backend != "feitcsi"
    monitor_note = ("not_needed_feitcsi_tunes_card"
                    if not needs_monitor_vif else "not_attempted")
    if (needs_monitor_vif and iface
            and not _iface_is_monitor(iface)):
        mon_cmds = _monitor_cmds(iface, channel)
        if all(common.can_run_priv(c) for c in mon_cmds):
            # root, or scoped passwordless sudo for iw/ip -> set monitor + channel.
            for cmd in mon_cmds:
                rc, _, err = common.run_priv(cmd)
                if rc != 0:
                    return common.emit(
                        AGENT, "start", False, node=csi["node"], iface=iface,
                        error=f"monitor setup failed: {' '.join(common.priv_cmd(cmd))}: "
                              f"{err.strip()}",
                        privileged_commands=[common.privileged_hint(c) for c in mon_cmds],
                    )
            monitor_note = (("by_root" if common.is_root() else "by_sudo")
                            if _iface_is_monitor(iface) else "incomplete")
        else:
            monitor_note = "needs_privilege"

    res = _delegate(
        csi_picoscenes.start,
        participant=args.participant, style=args.style, trial=args.trial,
        perspective=perspective, data_root=_data_root(),
        mon_iface=(iface or "mon_csi"), phy=csi.get("phy", "<phy>"),
    )
    return common.emit(
        AGENT, "start", bool(res.get("ok")), node=csi["node"], iface=iface,
        role=ROLE, perspective=perspective, channel=channel, width_mhz=WIDTH_MHZ,
        csi_backend=backend, monitor_setup=monitor_note, mode=res.get("mode"),
        out=res.get("out_raw"), capture=res,
    )


def stop(args) -> dict:
    from capture import csi_picoscenes

    csi = common.resolve_role_iface(ROLE, args.node)
    perspective = args.perspective or csi.get("perspective") or 1
    if not (args.participant and args.style and args.trial):
        return common.emit(AGENT, "stop", False,
                           error="participant/style/trial required")
    res = _delegate(
        csi_picoscenes.stop, participant=args.participant, style=args.style,
        trial=args.trial, perspective=perspective, data_root=_data_root(),
    )
    return common.emit(AGENT, "stop", bool(res.get("ok")), node=csi["node"],
                       mode=res.get("mode"), capture=res)


def status(args) -> dict:
    out_dir = common.default_out_dir(args)
    csi = common.resolve_role_iface(ROLE, args.node)
    info = common.read_pidfile(out_dir, AGENT) if out_dir else None
    running = bool(info and common.pid_alive(int(info.get("pid", -1))))
    return common.emit(
        AGENT, "status", True, node=csi["node"], iface=csi["iface"],
        running=running, pid=(info or {}).get("pid"),
        iface_type=common._iw_dev_types().get(csi["iface"], "unknown"),
        operstate=common._operstate(csi["iface"]) if csi["iface"] else "n/a",
        out=(info or {}).get("out"),
    )


def health(args) -> dict:
    csi = common.resolve_role_iface(ROLE, args.node)
    return common.emit(
        AGENT, "health", True, node=csi["node"],
        role=ROLE, csi_radio=csi,
        kernel=common.kernel_info(),
        clock=common.clock_sources(),
        tools=common.tool_availability(),
        radios=common.detect_radios(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m nodes.csi_agent")
    common.add_common_args(parser)
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
