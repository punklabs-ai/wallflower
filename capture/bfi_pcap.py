"""capture.bfi_pcap — thin pcap wrapper for IEEE 802.11 BFI capture.

Passively records the compressed beamforming reports the BFId paper attacks
(Todt, Morsbach, Strufe, CCS '25, sec. 3/5) into a single central
``bfi_recorder.pcapng`` on a monitor-mode AX210 radio tuned to the lab AP-BFI
channel (operative pilot: channel 36, 5 GHz, 80 MHz). Decoded later by
``parsers/parse_bfi.py``.

Tool preference: **tcpdump** (present on this node) writes the pcapng; tshark is
optional and only used for detection/inspection if available.

802.11 frames parse_bfi.py decodes
----------------------------------
The beamforming feedback rides in 802.11 management *action* frames:
  * VHT compressed beamforming  (Action category 21 / VHT, action 0)
  * HE compressed beamforming   (Action category 30 / HE, action 0)
The compressed report carries the quantised beamforming angles (psi/phi); the
paper uses 10 quantised angles x 74 channels => 740 BFI features. parse_bfi.py
walks the pcapng, filters these action frames for the target BSSID, dequantises
the angles and appends a per-frame time-delta column.

Capture targeting
-----------------
``start`` requires an explicit target BSSID and channel and refuses wildcard/
promiscuous capture. The tcpdump filter is pinned to the AP BSSID so only
beamforming feedback addressed to/from that AP is recorded.

stdlib-only; runs on a bare node. Real capture requires tcpdump and root (or
scoped passwordless sudo); with neither it writes no file and returns the
operator commands to run.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from wallflower.contract import (
    AP_BFI_CHANNEL,
    BAND_GHZ,
    WIDTH_MHZ,
)
from capture.naming import bfi_recorder_name, logs_dir, raw_trial_dir

AGENT = "bfi_pcap"
TCPDUMP_BIN = "tcpdump"
TSHARK_BIN = "tshark"

# Loose BSSID sanity check (colon-separated MAC). Refuse anything else so we
# can't be pointed at a wildcard/garbage target.
_BSSID_RE = re.compile(r"^[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_id() -> str:
    return os.uname().nodename


def _have_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


# --- scoped passwordless sudo (mirrors nodes.common; capture/ stays independent) -
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
    """Login user to hand the capture to (tcpdump -Z) so stop() can kill it."""
    import getpass
    return os.environ.get("SUDO_USER") or getpass.getuser()


def _child_pid(parent: int, name: str, tries: int = 20) -> int | None:
    """Find a child process of `parent` whose comm matches `name` (via /proc).

    When tcpdump is launched through `sudo`, the long-lived worker is sudo's
    child; after its `-Z` privilege drop it is owned by the invoking user and is
    the pid we must record so a later, unprivileged stop() can signal it.
    """
    proc_root = Path("/proc")
    for _ in range(tries):
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                comm = (entry / "comm").read_text().strip()
                if comm != name:
                    continue
                stat = (entry / "stat").read_text()
                ppid = int(stat.rsplit(")", 1)[1].split()[1])
                if ppid == parent:
                    return int(entry.name)
            except (OSError, ValueError, IndexError):
                continue
        time.sleep(0.05)
    return None


def tcpdump_path() -> str | None:
    return shutil.which(TCPDUMP_BIN)


def tshark_path() -> str | None:
    return shutil.which(TSHARK_BIN)


def available() -> dict:
    """Detection result. Never raises; safe unprivileged."""
    td = tcpdump_path()
    ts = tshark_path()
    return {
        "agent": AGENT,
        "action": "detect",
        "ok": td is not None,          # tcpdump is the required backend
        "node": _node_id(),
        "ts_utc": _now_iso(),
        "tcpdump": td,
        "tcpdump_available": td is not None,
        "tshark": ts,
        "tshark_available": ts is not None,
        "have_root": _have_root(),
    }


def _pidfile_path(trial_dir: Path) -> Path:
    return logs_dir(trial_dir) / f"{AGENT}.pid"


def validate_bssid(bssid: str) -> str:
    """Reject wildcard/empty/malformed BSSIDs."""
    if not bssid or not _BSSID_RE.match(bssid):
        raise ValueError(
            f"refusing capture: target BSSID {bssid!r} is missing or not a valid "
            "MAC. This wrapper captures from a single known AP only (no wildcard / "
            "promiscuous capture).")
    if bssid.lower() in ("ff:ff:ff:ff:ff:ff", "00:00:00:00:00:00"):
        raise ValueError("refusing broadcast/zero BSSID wildcard capture.")
    return bssid


def build_tcpdump_cmd(mon_iface: str, bssid: str, out_pcap: Path,
                      drop_user: str | None = None) -> list[str]:
    """Assemble the BSSID-pinned tcpdump command.

    Filters on the target AP BSSID and 802.11 management frames so only the
    beamforming-feedback exchange with the lab AP is recorded. ``-w`` writes
    pcapng (tcpdump writes pcapng with modern libpcap). No promiscuous wildcard.

    ``drop_user``: pass a username to add ``-Z <user>`` so tcpdump opens the
    interface as root then drops privileges to that user — the running worker is
    then owned by that user and a later unprivileged ``stop()`` can signal it.
    """
    validate_bssid(bssid)
    # BPF: management subtype frames involving the AP BSSID. Action frames are
    # management type; parse_bfi.py further narrows to VHT/HE compressed BF.
    bpf = f"wlan addr3 {bssid} and type mgt"
    cmd = [TCPDUMP_BIN, "-i", mon_iface, "-w", str(out_pcap), "-U"]
    if drop_user:
        cmd += ["-Z", drop_user]   # privilege drop so the worker is killable
    cmd.append(bpf)
    return cmd


def monitor_setup_cmds(mon_iface: str, phy: str = "<phy>") -> list[str]:
    """Privileged commands to ready the BFI monitor radio (printed, not run)."""
    return [
        f"sudo iw phy {phy} interface add {mon_iface} type monitor",
        f"sudo ip link set {mon_iface} up",
        f"sudo iw dev {mon_iface} set freq 5180 80 5210  "
        f"# AP-BFI ch{AP_BFI_CHANNEL} / {BAND_GHZ} GHz / {WIDTH_MHZ} MHz",
    ]


# --------------------------------------------------------------------------- #
# start / stop / status
# --------------------------------------------------------------------------- #
def start(participant: str, style: str, trial: str,
          *, bssid: str | None = None, data_root: Path | str = "data",
          mon_iface: str = "mon_bfi", phy: str = "<phy>") -> dict:
    """Begin central BFI pcap capture for one trial.

    Requires an explicit target AP ``bssid`` and tcpdump.
    Real path (tcpdump present, root): Popen tcpdump + pidfile. If tcpdump is
    missing or privilege is unavailable, returns ``ok=False`` with the operator
    commands to run — no capture file is written. Never raises; always returns
    structured JSON.
    """
    trial_dir = raw_trial_dir(data_root, participant, style, trial)
    ldir = logs_dir(trial_dir)
    ldir.mkdir(parents=True, exist_ok=True)
    out_pcap = trial_dir / bfi_recorder_name()
    pidfile = _pidfile_path(trial_dir)

    td = tcpdump_path()
    result: dict = {
        "agent": AGENT,
        "action": "start",
        "node": _node_id(),
        "ts_utc": _now_iso(),
        "participant": participant,
        "style": style,
        "trial": trial,
        "modality": "bfi",
        "out_pcap": str(out_pcap),
        "mon_iface": mon_iface,
        "channel": AP_BFI_CHANNEL,
        "band_ghz": BAND_GHZ,
        "width_mhz": WIDTH_MHZ,
        "bssid": bssid,
        "tcpdump_available": td is not None,
    }

    use_real = td is not None

    if use_real:
        # Real capture must target a known AP BSSID.
        try:
            validate_bssid(bssid or "")
        except ValueError as exc:
            result.update(ok=False, mode="refused", reason=str(exc))
            print(json.dumps(result))
            return result

        drop_user = None if _have_root() else _invoking_user()
        cmd = build_tcpdump_cmd(mon_iface, bssid, out_pcap, drop_user)  # type: ignore[arg-type]
        setup = monitor_setup_cmds(mon_iface, phy)
        result["tcpdump_cmd"] = cmd

        if not _can_priv(TCPDUMP_BIN):
            # Not root and no scoped passwordless sudo for tcpdump -> surface cmds.
            result.update(
                ok=False,
                mode="needs_privilege",
                reason="monitor mode + raw pcap capture require root or scoped "
                       "passwordless sudo for tcpdump; neither available.",
                operator_commands=setup + [" ".join(cmd)],
            )
            (ldir / f"{AGENT}.NEEDS_PRIVILEGE").write_text(
                "\n".join(setup + [" ".join(cmd)]) + "\n", encoding="utf-8")
            print(json.dumps(result))
            return result

        launch = _priv_wrap(cmd)         # `sudo -n tcpdump ...` unless already root
        via_sudo = launch[:2] == ["sudo", "-n"]
        result["launch_cmd"] = launch
        try:
            stderr_fp = (ldir / f"{AGENT}.stderr").open("ab")
            proc = subprocess.Popen(
                launch, stdout=subprocess.DEVNULL, stderr=stderr_fp,
                start_new_session=True)
            # When launched via sudo, proc.pid is sudo (root). Record the tcpdump
            # WORKER pid instead — after `-Z` it is owned by drop_user, so the
            # unprivileged stop() can signal it (killing it makes sudo exit too).
            worker = _child_pid(proc.pid, "tcpdump") if via_sudo else proc.pid
            rec_pid = worker if worker is not None else proc.pid
            pidfile.write_text(str(rec_pid), encoding="utf-8")
            result.update(ok=True,
                          mode="tcpdump_sudo" if via_sudo else "tcpdump",
                          pid=rec_pid, launcher_pid=proc.pid,
                          drop_user=drop_user, pidfile=str(pidfile))
        except OSError as exc:
            result.update(ok=False, mode="error", reason=f"launch failed: {exc}")
        print(json.dumps(result))
        return result

    # tcpdump missing: no capture is written. Surface the operator commands.
    if bssid:
        try:
            validate_bssid(bssid)
        except ValueError as exc:
            result.update(ok=False, mode="refused", reason=str(exc))
            print(json.dumps(result))
            return result

    setup = monitor_setup_cmds(mon_iface, phy)
    real_cmd = (build_tcpdump_cmd(mon_iface, bssid, out_pcap)
                if bssid else
                [TCPDUMP_BIN, "-i", mon_iface, "-w", str(out_pcap), "-U",
                 "wlan addr3 <LAB_AP_BSSID> and type mgt"])
    note = ldir / f"{AGENT}.MISSING_TCPDUMP"
    note.write_text(
        "tcpdump not installed; no capture written.\n"
        "To capture for real (root, known lab AP BSSID) run:\n"
        + "\n".join(setup) + "\n" + " ".join(real_cmd) + "\n",
        encoding="utf-8")
    result.update(
        ok=False,
        mode="needs_tcpdump",
        reason="tcpdump is not installed; real BFI capture is not possible.",
        note=str(note),
        operator_commands=setup + [" ".join(real_cmd)],
    )
    print(json.dumps(result))
    return result


def stop(participant: str, style: str, trial: str,
         *, data_root: Path | str = "data") -> dict:
    """Stop a previously started BFI capture by pidfile (SIGTERM)."""
    trial_dir = raw_trial_dir(data_root, participant, style, trial)
    pidfile = _pidfile_path(trial_dir)

    result: dict = {
        "agent": AGENT,
        "action": "stop",
        "node": _node_id(),
        "ts_utc": _now_iso(),
        "participant": participant,
        "style": style,
        "trial": trial,
        "modality": "bfi",
        "pidfile": str(pidfile),
    }

    if not pidfile.exists():
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


def status(participant: str, style: str, trial: str,
           *, data_root: Path | str = "data") -> dict:
    """Report whether the BFI capture pidfile/process is alive + artifact size."""
    trial_dir = raw_trial_dir(data_root, participant, style, trial)
    pidfile = _pidfile_path(trial_dir)
    out_pcap = trial_dir / bfi_recorder_name()

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
        "modality": "bfi",
        "pidfile": str(pidfile),
        "pid": pid,
        "running": alive,
        "out_pcap": str(out_pcap),
        "out_pcap_exists": out_pcap.exists(),
        "out_pcap_bytes": out_pcap.stat().st_size if out_pcap.exists() else 0,
    }
    print(json.dumps(result))
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"capture.{AGENT}",
        description="Thin tcpdump BFI pcap wrapper (BSSID-pinned, no wildcard "
                    "capture).")
    p.add_argument("action", choices=["detect", "start", "stop", "status"])
    p.add_argument("--participant")
    p.add_argument("--style")
    p.add_argument("--trial")
    p.add_argument("--bssid", help="target AP BSSID (required for real capture)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--mon-iface", default="mon_bfi",
                   help="monitor interface name on the BFI radio")
    p.add_argument("--phy", default="<phy>",
                   help="phy of the BFI radio (for printed setup commands)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.action == "detect":
        print(json.dumps(available()))
        return 0

    missing = [k for k in ("participant", "style", "trial")
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
        res = start(args.participant, args.style, args.trial,
                    bssid=args.bssid, mon_iface=args.mon_iface, phy=args.phy, **kw)
    elif args.action == "stop":
        res = stop(args.participant, args.style, args.trial, **kw)
    else:  # status
        res = status(args.participant, args.style, args.trial, **kw)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
