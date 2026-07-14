"""capture.csi_picoscenes — thin wrapper around an AX210 CSI tool for capture.

Records per-perspective CSI on an AX210 radio tuned to the lab AP-CSI channel
(operative pilot: channel 36, 5 GHz, 80 MHz) and writes a deterministic
``csi_p<perspective>.raw`` into the trial directory. Parsed later by
``parsers/parse_csi.py``.

Backend choice (see Research CSI-TOOL / CSI-FORMAT)
---------------------------------------------------
The reference AX210 CSI toolchains are **FeitCSI** (open source, github.com/
KuskoSoft/FeitCSI) and **PicoScenes** (https://ps.zpj.io, free for academia).
On node1 (Ubuntu 26.04, kernel 7.0, 2x AX210) BOTH are **DRIVER-GATED**: neither
ships a kernel-7.0 driver, and the driver would REPLACE the live in-tree iwlwifi
/ mac80211 / cfg80211 stack that node1's 6 GHz link + RSSI dashboard depend on.
So real CSI is **operator-gated and risk-accepted**, never enabled by this agent.

We therefore prefer **FeitCSI** as the real backend (it tunes the card itself,
no separate monitor-VIF dance) and keep **PicoScenes** as a secondary. This
module:

  * detects availability of the ``feitcsi`` / ``PicoScenes`` binaries (``which``);
  * ``start(...)`` launches the present tool via Popen + writes a pidfile; when no
    tool or privilege is available it writes no file and returns ``ok=False`` with
    the operator commands to run;
  * ``stop(...)`` terminates a previously started capture by pidfile.

stdlib-only; no numpy/torch (this must run on a bare node).

Expected real invocation (documented, not yet runnable here)
------------------------------------------------------------
**FeitCSI (preferred)** tunes the *card* directly — no monitor VIF, but it drives
the radio so it conflicts with a managed association; capture must use the AX210
that is NOT carrying the dashboard monitor. Active control is 5180 MHz @ 80 MHz::

    feitcsi --frequency 5180 --channel-width 80 --format HESU \
            --output-file <trial_dir>/csi_p<perspective>.raw -v

**PicoScenes (secondary)** captures from a monitor interface created on the CSI
radio; the operator first (with root / scoped sudo for iw+ip) puts the radio in
monitor and sets the configured frequency. ``start`` prints those exact privileged
commands when run unprivileged. The capture command is roughly::

    PicoScenes "-d info; -i <mon_iface> --mode logger
        --freq 6 --channel 37 --bw 160 --output <trial_dir>/csi_p<perspective>.raw"

(Exact flag spelling varies by build; both are assembled below and logged so the
operator can copy/adjust them.)

CRITICAL OPERATOR NOTE: installing/loading a CSI driver (insmod/modprobe/dkms)
is NOT in the scoped sudo allow-list (only iw/ip/tcpdump/apt-get/nmcli) and MUST
NOT be attempted here; it would drop node1's live 6 GHz link + dashboard. The
real-capture path only ever launches the *userspace* tool if it is already
installed; it never installs/loads a driver.

Recorded ``.raw`` format
------------------------
The real artifact is the CSI tool's native frame log; parse_csi.py branches on
the tool's native signature to decide how to decode it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from wallflower.contract import (
    AP_CSI_CHANNEL,
    BAND_GHZ,
    WIDTH_MHZ,
    monitor_freq_args,
    validate_perspective,
)
from capture.naming import csi_raw_name, logs_dir, raw_trial_dir

AGENT = "csi_picoscenes"
# Preferred real backend is FeitCSI (open, kernel-7.0-plausible, card-tuning);
# PicoScenes kept as a secondary backend.
FEITCSI_BIN = "feitcsi"
PICOSCENES_BIN = "PicoScenes"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_id() -> str:
    return os.uname().nodename


def _have_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


# --- scoped passwordless sudo (mirrors nodes.common; capture/ stays independent) -
# capture/ must not import nodes/, so we re-implement the same NOPASSWD parsing
# pattern used in capture/bfi_pcap.py.
def _nopasswd_binaries() -> list[str]:
    """Absolute paths runnable via passwordless `sudo -n` (scoped sudoers)."""
    if not shutil.which("sudo"):
        return []
    try:
        p = subprocess.run(["sudo", "-n", "-l"], capture_output=True,
                           text=True, timeout=5)
    except Exception:
        return []
    if p.returncode != 0:
        return []
    found: list[str] = []
    for line in p.stdout.splitlines():
        if "NOPASSWD:" in line:
            for tok in line.split("NOPASSWD:", 1)[1].split(","):
                tok = tok.strip()
                if tok:
                    tok = tok.split()[0]
                if tok.startswith("/"):
                    found.append(tok)
    return found


def _can_priv(binary: str) -> bool:
    """True if `binary` can run privileged now (root, or scoped passwordless sudo)."""
    if _have_root():
        return True
    return (shutil.which(binary) or binary) in _nopasswd_binaries()


def _priv_wrap(cmd: list[str]) -> list[str]:
    """Prefix `sudo -n` when not root and the binary is on the NOPASSWD allow-list."""
    if _have_root() or not cmd:
        return list(cmd)
    if _can_priv(cmd[0]):
        return ["sudo", "-n", *cmd]
    return list(cmd)


def _invoking_user() -> str:
    """Login user to hand a privilege-dropping capture to, so stop() can kill it."""
    import getpass
    return os.environ.get("SUDO_USER") or getpass.getuser()


def feitcsi_path() -> str | None:
    """Absolute path to the FeitCSI binary, or None if not installed."""
    return shutil.which(FEITCSI_BIN)


def picoscenes_path() -> str | None:
    """Absolute path to the PicoScenes binary, or None if not installed."""
    return shutil.which(PICOSCENES_BIN)


def csi_tool() -> tuple[str | None, str | None]:
    """Return (backend_name, binary_path) for the first available CSI tool.

    Preference order: FeitCSI (recommended) then PicoScenes. ``(None, None)`` if
    neither is installed.
    """
    fc = feitcsi_path()
    if fc is not None:
        return "feitcsi", fc
    ps = picoscenes_path()
    if ps is not None:
        return "picoscenes", ps
    return None, None


def available() -> dict:
    """Detection result. Never raises; safe to call unprivileged."""
    fc = feitcsi_path()
    ps = picoscenes_path()
    backend, path = csi_tool()
    return {
        "agent": AGENT,
        "action": "detect",
        "ok": True,
        "node": _node_id(),
        "ts_utc": _now_iso(),
        "feitcsi": fc,
        "feitcsi_available": fc is not None,
        "picoscenes": ps,
        "picoscenes_available": ps is not None,
        "csi_backend": backend,
        "csi_tool": path,
        "csi_tool_available": path is not None,
        "have_root": _have_root(),
    }


def _pidfile_path(trial_dir: Path, perspective: int) -> Path:
    return logs_dir(trial_dir) / f"{AGENT}_p{perspective}.pid"


# --------------------------------------------------------------------------- #
# Real CSI command builders
# --------------------------------------------------------------------------- #
def build_feitcsi_cmd(out_raw: Path, perspective: int) -> list[str]:
    """Assemble the local FeitCSI capture command for the configured AP-CSI.

    FeitCSI tunes the card directly (no monitor VIF). ``HESU`` selects the
    802.11ax single-user format the AP-CSI link uses.
    Flag spelling may need adjusting per FeitCSI build; the command is always
    logged so the operator can verify/copy it.
    """
    validate_perspective(perspective)
    control = monitor_freq_args(AP_CSI_CHANNEL, BAND_GHZ)[0]
    return [
        FEITCSI_BIN,
        "--frequency", str(control),
        "--channel-width", str(WIDTH_MHZ),
        "--format", "HESU",
        "--output-file", str(out_raw),
        "-v",
    ]


def build_picoscenes_cmd(mon_iface: str, out_raw: Path, perspective: int) -> list[str]:
    """Assemble the documented PicoScenes logger command line.

    Logs CSI from a monitor interface (created/tuned by the operator/agent via
    iw+ip) on the configured AP-CSI channel to ``out_raw``. Flag
    spelling may need adjusting per PicoScenes build; the command is always logged
    so the operator can verify/copy it.
    """
    validate_perspective(perspective)
    return [
        PICOSCENES_BIN,
        "-d", "info",
        "-i", mon_iface,
        "--mode", "logger",
        "--freq", str(BAND_GHZ),
        "--channel", str(AP_CSI_CHANNEL),
        "--bw", str(WIDTH_MHZ),
        "--output", str(out_raw),
    ]


def build_csi_cmd(backend: str, mon_iface: str, out_raw: Path,
                  perspective: int) -> list[str]:
    """Dispatch to the backend-specific command builder."""
    if backend == "feitcsi":
        return build_feitcsi_cmd(out_raw, perspective)
    if backend == "picoscenes":
        return build_picoscenes_cmd(mon_iface, out_raw, perspective)
    raise ValueError(f"unknown CSI backend {backend!r}")


def monitor_setup_cmds(mon_iface: str, phy: str = "<phy>", *,
                       backend: str | None = None) -> list[str]:
    """Privileged commands the operator must run to ready the CSI radio.

    Printed (never executed here) so the operator can copy/paste them. We tune
    by FREQUENCY in both bands so channel numbers stay unambiguous.

    For **FeitCSI** the card is tuned by the tool itself, so no monitor VIF is
    needed; we only note that the radio must NOT be the dashboard monitor radio and
    that the FeitCSI driver must already be installed by the operator (NOT here).
    For **PicoScenes** the operator creates+tunes a monitor VIF first.
    """
    control, width, center = monitor_freq_args(AP_CSI_CHANNEL, BAND_GHZ)
    if backend == "feitcsi":
        return [
            "sudo iw reg set US",
            f"# FeitCSI tunes the card directly to {control} MHz @ {width} "
            f"MHz (ch{AP_CSI_CHANNEL}); use the AX210 NOT carrying the dashboard "
            "monitor radio. The FeitCSI driver must be installed by the operator "
            "(dkms/insmod are NOT passwordless and must NOT be run by the agent).",
        ]
    # PicoScenes (or unknown): create + tune a monitor VIF on the CSI radio.
    return [
        "sudo iw reg set US",
        f"sudo iw phy {phy} interface add {mon_iface} type monitor",
        f"sudo ip link set {mon_iface} up",
        f"sudo iw dev {mon_iface} set freq {control} {width} {center}  "
        f"# AP-CSI ch{AP_CSI_CHANNEL} / {BAND_GHZ} GHz / {WIDTH_MHZ} MHz",
    ]


# --------------------------------------------------------------------------- #
# start / stop
# --------------------------------------------------------------------------- #
def start(participant: str, style: str, trial: str, perspective: int,
          *, data_root: Path | str = "data", mon_iface: str = "mon_csi",
          phy: str = "<phy>") -> dict:
    """Begin CSI capture for one perspective.

    Real path (a CSI tool present + runnable): Popen the tool and write a
    pidfile. When no tool or privilege is available it writes no file and returns
    ``ok=False`` with the operator commands to run. Never raises on missing
    tools/privilege; always returns a structured JSON dict.

    The real path NEVER installs or loads a driver — it only launches the
    *userspace* CSI tool if it is already installed (driver install is operator-
    gated; see module docstring).
    """
    perspective = validate_perspective(perspective)
    trial_dir = raw_trial_dir(data_root, participant, style, trial)
    ldir = logs_dir(trial_dir)
    ldir.mkdir(parents=True, exist_ok=True)
    out_raw = trial_dir / csi_raw_name(perspective)
    pidfile = _pidfile_path(trial_dir, perspective)

    backend, tool_path = csi_tool()
    cmd = (build_csi_cmd(backend, mon_iface, out_raw, perspective)
           if backend else
           # No tool: still surface the recommended (FeitCSI) command for the op.
           build_feitcsi_cmd(out_raw, perspective))
    setup = monitor_setup_cmds(mon_iface, phy, backend=backend or "feitcsi")

    result: dict = {
        "agent": AGENT,
        "action": "start",
        "node": _node_id(),
        "ts_utc": _now_iso(),
        "participant": participant,
        "style": style,
        "trial": trial,
        "perspective": perspective,
        "modality": "csi",
        "out_raw": str(out_raw),
        "mon_iface": mon_iface,
        "channel": AP_CSI_CHANNEL,
        "band_ghz": BAND_GHZ,
        "width_mhz": WIDTH_MHZ,
        "csi_backend": backend,
        "csi_cmd": cmd,
        "feitcsi_available": feitcsi_path() is not None,
        "picoscenes_available": picoscenes_path() is not None,
        "csi_tool_available": tool_path is not None,
    }

    use_real = backend is not None

    if use_real:
        # FeitCSI tunes the card itself and accesses the netlink/cfg80211 control
        # path -> needs root (or scoped passwordless sudo for the tool, which is
        # NOT on the allow-list). PicoScenes monitor capture likewise needs root.
        # We mirror bfi_pcap: launch via `sudo -n` only if the tool is on the
        # NOPASSWD allow-list (it isn't on this node), else surface the commands.
        if not _can_priv(tool_path):
            result.update(
                ok=False,
                mode="needs_privilege",
                reason=f"{backend} CSI capture requires root (or scoped "
                       "passwordless sudo for the tool, which is not granted); "
                       "not attempting to escalate or load a driver.",
                operator_commands=setup + [" ".join(cmd)],
            )
            (ldir / f"{AGENT}_p{perspective}.NEEDS_PRIVILEGE").write_text(
                "\n".join(setup + [" ".join(cmd)]) + "\n", encoding="utf-8")
            print(json.dumps(result))
            return result

        launch = _priv_wrap(cmd)        # `sudo -n <tool> ...` unless already root
        via_sudo = launch[:2] == ["sudo", "-n"]
        result["launch_cmd"] = launch
        try:
            stderr_fp = (ldir / f"{AGENT}_p{perspective}.stderr").open("ab")
            proc = subprocess.Popen(
                launch, stdout=subprocess.DEVNULL, stderr=stderr_fp,
                start_new_session=True)
            pidfile.write_text(str(proc.pid), encoding="utf-8")
            result.update(ok=True,
                          mode=f"{backend}_sudo" if via_sudo else backend,
                          pid=proc.pid, launcher_pid=proc.pid,
                          pidfile=str(pidfile))
        except OSError as exc:  # binary vanished / exec failure
            result.update(ok=False, mode="error", reason=f"launch failed: {exc}")
        print(json.dumps(result))
        return result

    # No CSI tool installed: no capture is written. Surface the operator commands.
    note = ldir / f"{AGENT}_p{perspective}.MISSING_CSI"
    note.write_text(
        "no CSI tool installed (FeitCSI/PicoScenes); no capture written.\n"
        "Real CSI is DRIVER-GATED on this node (kernel 7.0, AX210): the operator "
        "must install a CSI driver out-of-band (NOT via this agent). To capture "
        "for real, install FeitCSI then run:\n"
        + "\n".join(setup) + "\n" + " ".join(cmd) + "\n",
        encoding="utf-8")
    result.update(
        ok=False,
        mode="needs_csi_tool",
        reason="no CSI tool installed (FeitCSI/PicoScenes); real CSI capture is "
               "not possible.",
        note=str(note),
        operator_commands=setup + [" ".join(cmd)],
    )
    print(json.dumps(result))
    return result


def stop(participant: str, style: str, trial: str, perspective: int,
         *, data_root: Path | str = "data") -> dict:
    """Stop a previously started CSI capture by pidfile (SIGTERM)."""
    perspective = validate_perspective(perspective)
    trial_dir = raw_trial_dir(data_root, participant, style, trial)
    pidfile = _pidfile_path(trial_dir, perspective)
    out_raw = trial_dir / csi_raw_name(perspective)

    result: dict = {
        "agent": AGENT,
        "action": "stop",
        "node": _node_id(),
        "ts_utc": _now_iso(),
        "participant": participant,
        "style": style,
        "trial": trial,
        "perspective": perspective,
        "modality": "csi",
        "pidfile": str(pidfile),
    }

    if not pidfile.exists():
        # No pidfile means nothing was launched here: that's a clean stop.
        result.update(ok=True, mode="no_pidfile",
                      reason="no live capture (never started or already stopped)")
        print(json.dumps(result))
        return result

    try:
        pid = int(pidfile.read_text(encoding="utf-8").strip())
    except (ValueError, OSError) as exc:
        result.update(ok=False, mode="error", reason=f"bad pidfile: {exc}")
        print(json.dumps(result))
        return result

    try:
        os.kill(pid, signal.SIGTERM)
        result.update(ok=True, mode="terminated", pid=pid)
    except ProcessLookupError:
        result.update(ok=True, mode="already_dead", pid=pid)
    except PermissionError:
        result.update(ok=False, mode="needs_privilege", pid=pid,
                      reason="cannot signal pid (owned by root?)",
                      operator_commands=[f"sudo kill {pid}"])
    finally:
        try:
            pidfile.unlink()
        except OSError:
            pass
    print(json.dumps(result))
    return result


def status(participant: str, style: str, trial: str, perspective: int,
           *, data_root: Path | str = "data") -> dict:
    """Report whether a CSI capture pidfile exists and the process is alive."""
    perspective = validate_perspective(perspective)
    trial_dir = raw_trial_dir(data_root, participant, style, trial)
    pidfile = _pidfile_path(trial_dir, perspective)
    out_raw = trial_dir / csi_raw_name(perspective)

    alive = False
    pid: int | None = None
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            alive = True
        except (ValueError, OSError):
            alive = False

    result = {
        "agent": AGENT,
        "action": "status",
        "ok": True,
        "node": _node_id(),
        "ts_utc": _now_iso(),
        "participant": participant,
        "style": style,
        "trial": trial,
        "perspective": perspective,
        "modality": "csi",
        "pidfile": str(pidfile),
        "pid": pid,
        "running": alive,
        "out_raw": str(out_raw),
        "out_raw_exists": out_raw.exists(),
        "out_raw_bytes": out_raw.stat().st_size if out_raw.exists() else 0,
    }
    print(json.dumps(result))
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"capture.{AGENT}",
        description="Thin AX210 CSI capture wrapper (FeitCSI/PicoScenes).")
    p.add_argument("action", choices=["detect", "start", "stop", "status"])
    p.add_argument("--participant")
    p.add_argument("--style")
    p.add_argument("--trial")
    p.add_argument("--perspective", type=int)
    p.add_argument("--data-root", default="data")
    p.add_argument("--mon-iface", default="mon_csi",
                   help="monitor interface name on the CSI radio (PicoScenes)")
    p.add_argument("--phy", default="<phy>",
                   help="phy of the CSI radio (for printed setup commands)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.action == "detect":
        print(json.dumps(available()))
        return 0

    missing = [k for k in ("participant", "style", "trial", "perspective")
               if getattr(args, k) is None]
    if missing:
        err = {
            "agent": AGENT, "action": args.action, "ok": False,
            "node": _node_id(), "ts_utc": _now_iso(),
            "reason": f"missing required args: {', '.join(missing)}",
        }
        print(json.dumps(err))
        return 2

    kw = dict(data_root=args.data_root)
    if args.action == "start":
        res = start(args.participant, args.style, args.trial, args.perspective,
                    mon_iface=args.mon_iface, phy=args.phy, **kw)
    elif args.action == "stop":
        res = stop(args.participant, args.style, args.trial, args.perspective, **kw)
    else:  # status
        res = status(args.participant, args.style, args.trial, args.perspective, **kw)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
