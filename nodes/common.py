"""Shared, stdlib-only helpers for wallflower node agents.

Factors out the concerns every node agent needs:
  * AX210 interface detection (parse `iw dev` + /sys/class/net/*/phy80211 + lspci)
  * physical-radio -> logical-role mapping (configs/nodes.yaml else iface name)
  * driver/kernel/firmware/iface-state reporting (/sys, `iw dev`, `ethtool -i`,
    `uname`)
  * a tiny YAML-subset loader (pyyaml is NOT assumed present on bare nodes)
  * structured JSON logging (print + return one dict)
  * privilege detection + "print the exact privileged command instead of
    crashing" behaviour
  * pidfile read/write/kill for the start/stop lifecycle

NO third-party imports here: node agents must run on a freshly-imaged node with
nothing but the Python 3 standard library. Numeric/path constants come from
wallflower.contract (imported lazily where needed so detection still works even if the
package is not on PYTHONPATH on a minimal node).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Repo / config locations
# --------------------------------------------------------------------------- #
# nodes/ is directly under the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
NODES_YAML = CONFIG_DIR / "nodes.yaml"
LAB_YAML = CONFIG_DIR / "lab.yaml"


# --------------------------------------------------------------------------- #
# Structured JSON logging
# --------------------------------------------------------------------------- #
def now_utc() -> str:
    """ISO-8601 UTC timestamp (seconds precision, 'Z' suffix)."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def hostname() -> str:
    try:
        return os.uname().nodename
    except Exception:
        return "unknown"


def emit(agent: str, action: str, ok: bool, **fields: Any) -> dict:
    """Build the one structured JSON log object, print it to stdout, return it.

    Always contains: agent, action, ok, node, ts_utc (NODE-AGENT contract).
    `node` defaults to the resolved logical node id if present in fields, else
    the OS hostname.
    """
    obj: dict[str, Any] = {
        "agent": agent,
        "action": action,
        "ok": bool(ok),
        "node": fields.pop("node", None) or hostname(),
        "ts_utc": now_utc(),
    }
    obj.update(fields)
    # One JSON object, one line, flushed (orchestrator parses stdout).
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()
    return obj


# --------------------------------------------------------------------------- #
# Tiny YAML-subset loader (no pyyaml on bare nodes)
# --------------------------------------------------------------------------- #
# Supports the limited subset used by configs/nodes.yaml + configs/lab.yaml:
#   * nested mappings via 2-space indentation
#   * "key: value" scalars (int/float/bool/null/str, quoted or bare)
#   * "key:" introducing a nested block (map or list)
#   * list items "- value" (scalar) or "- key: value" (start of a map item)
#   * inline flow lists "[a, b, c]"
#   * '#' comments and blank lines ignored
# This is intentionally minimal; it is NOT a general YAML parser.
def _coerce_scalar(tok: str) -> Any:
    s = tok.strip()
    if s == "" or s in ("~", "null", "Null", "NULL"):
        return None
    if (s[0] == s[-1]) and s[0] in ("'", '"') and len(s) >= 2:
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_coerce_scalar(p) for p in _split_flow(inner)]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _split_flow(inner: str) -> list[str]:
    """Split a flow-list body on commas not inside quotes."""
    out: list[str] = []
    buf = ""
    quote = ""
    for ch in inner:
        if quote:
            buf += ch
            if ch == quote:
                quote = ""
        elif ch in ("'", '"'):
            quote = ch
            buf += ch
        elif ch == ",":
            out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf)
    return out


def _strip_comment(line: str) -> str:
    """Remove a trailing # comment that is not inside quotes."""
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            return line[:i]
    return line


def load_yaml(path: Path | str) -> dict:
    """Parse the restricted YAML subset used by this repo's configs.

    Returns {} if the file is missing (callers degrade gracefully).
    """
    p = Path(path)
    if not p.exists():
        return {}
    raw_lines = p.read_text().splitlines()
    # Build (indent, content) tuples, dropping blanks/comments.
    items: list[tuple[int, str]] = []
    for line in raw_lines:
        no_comment = _strip_comment(line)
        if not no_comment.strip():
            continue
        indent = len(no_comment) - len(no_comment.lstrip(" "))
        items.append((indent, no_comment.strip()))
    pos = 0

    def parse_block(min_indent: int) -> Any:
        nonlocal pos
        # Decide list vs map by first item at this indent.
        if pos >= len(items):
            return None
        indent, content = items[pos]
        if content.startswith("- "):
            return parse_list(indent)
        return parse_map(min_indent)

    def parse_map(indent_level: int) -> dict:
        nonlocal pos
        result: dict[str, Any] = {}
        while pos < len(items):
            indent, content = items[pos]
            if indent < indent_level:
                break
            if indent > indent_level:
                # Should have been consumed by a nested parse; skip defensively.
                pos += 1
                continue
            if content.startswith("- "):
                break
            if ":" not in content:
                pos += 1
                continue
            key, _, rest = content.partition(":")
            key = key.strip()
            rest = rest.strip()
            pos += 1
            if rest == "":
                # Nested block follows (map or list) at deeper indent, else null.
                if pos < len(items) and items[pos][0] > indent:
                    result[key] = parse_block(items[pos][0])
                else:
                    result[key] = None
            else:
                result[key] = _coerce_scalar(rest)
        return result

    def parse_list(indent_level: int) -> list:
        nonlocal pos
        result: list[Any] = []
        while pos < len(items):
            indent, content = items[pos]
            if indent != indent_level or not content.startswith("- "):
                break
            body = content[2:].strip()
            if ":" in body and not (body.startswith("[") or body.startswith(("'", '"'))):
                # Map item: rewrite this line as a map entry at a virtual indent
                # and parse a map made of this + following deeper lines.
                key, _, rest = body.partition(":")
                # Replace current item so parse_map sees the first key at
                # indent_level+2 (the conventional "- " offset).
                items[pos] = (indent_level + 2, f"{key.strip()}: {rest.strip()}".rstrip())
                result.append(parse_map(indent_level + 2))
            else:
                result.append(_coerce_scalar(body))
                pos += 1
        return result

    parsed = parse_block(items[0][0]) if items else {}
    return parsed if isinstance(parsed, dict) else {"_root": parsed}


# --------------------------------------------------------------------------- #
# Command / privilege helpers
# --------------------------------------------------------------------------- #
def have_tool(name: str) -> bool:
    return shutil.which(name) is not None


def tool_path(name: str) -> str | None:
    return shutil.which(name)


def is_root() -> bool:
    try:
        return os.geteuid() == 0
    except AttributeError:  # pragma: no cover - non-posix
        return False


def privileged_hint(cmd: list[str] | str) -> str:
    """Format the EXACT command an operator should run as root.

    sudo on this lab requires a password (non-interactive root unavailable), so
    we never attempt to escalate; we surface the command for a human to run.
    """
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    return f"sudo {cmd}"


def run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a command, never raise. Returns (rc, stdout, stderr).

    rc == 127 means the binary was not found.
    """
    if not cmd or shutil.which(cmd[0]) is None:
        return 127, "", f"{cmd[0] if cmd else '?'}: not found"
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"
    except Exception as exc:  # pragma: no cover
        return 1, "", str(exc)


# --------------------------------------------------------------------------- #
# Privileged execution via scoped passwordless sudo
# --------------------------------------------------------------------------- #
# This lab node may carry a scoped sudoers drop-in (e.g. /etc/sudoers.d/wallflower)
# granting passwordless sudo for ONLY specific binaries (iw, ip, tcpdump,
# apt-get). We NEVER store or pass a password. We detect what `sudo -n` permits
# without a password by parsing the NOPASSWD lines of `sudo -n -l` (the plain
# `sudo -l <cmd>` check is unreliable when the user also has a broad password
# rule). When privilege is unavailable we still surface the exact command for a
# human, exactly as before — so behaviour degrades gracefully on locked-down nodes.
_NOPASSWD_CACHE: list[str] | None = None


def nopasswd_binaries() -> list[str]:
    """Absolute paths this user may run via `sudo -n` WITHOUT a password."""
    global _NOPASSWD_CACHE
    if _NOPASSWD_CACHE is not None:
        return _NOPASSWD_CACHE
    found: list[str] = []
    if shutil.which("sudo"):
        rc, out, _ = run(["sudo", "-n", "-l"], timeout=5.0)
        if rc == 0:
            for line in out.splitlines():
                if "NOPASSWD:" in line:
                    rhs = line.split("NOPASSWD:", 1)[1]
                    for tok in rhs.split(","):
                        tok = tok.strip()
                        if tok:
                            tok = tok.split()[0]  # drop any args
                        if tok.startswith("/"):
                            found.append(tok)
    _NOPASSWD_CACHE = found
    return found


def sudo_ok(binary: str) -> bool:
    """True if `sudo -n <binary>` runs without a password (scoped NOPASSWD)."""
    path = shutil.which(binary) or binary
    return path in nopasswd_binaries()


def can_run_priv(cmd: list[str]) -> bool:
    """True if `cmd` can run privileged right now (already root, or scoped sudo)."""
    return is_root() or (bool(cmd) and sudo_ok(cmd[0]))


def priv_cmd(cmd: list[str]) -> list[str]:
    """Wrap `cmd` to run privileged: unchanged if root, else `sudo -n <cmd>`.

    Only prefixes sudo when the binary is on the passwordless allow-list; callers
    should gate on :func:`can_run_priv` and otherwise surface
    :func:`privileged_hint` so an operator can run it manually.
    """
    if is_root() or not cmd:
        return list(cmd)
    if sudo_ok(cmd[0]):
        return ["sudo", "-n", *cmd]
    return list(cmd)


def run_priv(cmd: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run `cmd` with privilege if possible (root or scoped `sudo -n`).

    rc == 126 means privilege is unavailable (not root and no passwordless sudo
    for this binary); the caller should then surface :func:`privileged_hint`.
    """
    if not cmd:
        return 1, "", "empty command"
    if can_run_priv(cmd):
        return run(priv_cmd(cmd), timeout=timeout)
    return 126, "", (f"{cmd[0]}: requires privilege "
                     "(not root and sudo -n not permitted)")


# --------------------------------------------------------------------------- #
# Radio / interface detection
# --------------------------------------------------------------------------- #
SYS_NET = Path("/sys/class/net")


def _read_sys(path: Path) -> str:
    try:
        return path.read_text().strip()
    except Exception:
        return ""


def list_wireless_ifaces() -> list[str]:
    """Wireless netdev names via /sys/class/net/*/phy80211 (no root needed)."""
    out: list[str] = []
    if not SYS_NET.exists():
        return out
    for entry in sorted(SYS_NET.iterdir()):
        if (entry / "phy80211").exists():
            out.append(entry.name)
    return out


def _phy_for(iface: str) -> str:
    link = SYS_NET / iface / "phy80211"
    try:
        return os.path.basename(os.path.realpath(link))
    except Exception:
        return ""


def _pci_for(iface: str) -> str:
    """PCI bus id (e.g. '01:00.0') from /sys device symlink."""
    try:
        dev = os.path.realpath(SYS_NET / iface / "device")
        m = re.search(r"([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])$", dev)
        return m.group(1) if m else ""
    except Exception:
        return ""


def _mac_for(iface: str) -> str:
    return _read_sys(SYS_NET / iface / "address")


def _operstate(iface: str) -> str:
    return _read_sys(SYS_NET / iface / "operstate") or "unknown"


def _lspci_ax210() -> dict[str, str]:
    """Map PCI id -> description for AX210 controllers (read-only, no root)."""
    rc, out, _ = run(["lspci"])
    result: dict[str, str] = {}
    if rc != 0:
        return result
    for line in out.splitlines():
        if "AX210" in line or "AX1675" in line:
            pci = line.split()[0]
            # normalise to full domainless 'bb:dd.f'
            result[pci] = line.partition(" ")[2].strip()
    return result


def _iw_dev_types() -> dict[str, str]:
    """iface -> interface type ('managed'/'monitor'/...) from `iw dev`."""
    rc, out, _ = run(["iw", "dev"])
    types: dict[str, str] = {}
    cur = None
    if rc != 0:
        return types
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Interface "):
            cur = s.split(None, 1)[1]
        elif s.startswith("Unnamed/non-netdev interface"):
            cur = None
        elif s.startswith("type ") and cur:
            types[cur] = s.split(None, 1)[1]
    return types


def _ethtool_info(iface: str) -> dict[str, str]:
    rc, out, _ = run(["ethtool", "-i", iface])
    info: dict[str, str] = {}
    if rc != 0:
        return info
    for line in out.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            info[k.strip()] = v.strip()
    return info


def detect_radios() -> list[dict]:
    """Detect AX210 radios on this host (read-only, root NOT required).

    Returns a list of dicts: {iface, phy, pci, mac, operstate, type, is_ax210,
    driver, firmware}. Combines /sys, `iw dev`, `lspci`, `ethtool -i`.
    """
    ax_pci = _lspci_ax210()
    types = _iw_dev_types()
    radios: list[dict] = []
    for iface in list_wireless_ifaces():
        pci = _pci_for(iface)
        eth = _ethtool_info(iface)
        is_ax210 = (pci in ax_pci) or ("AX210" in str(eth.get("firmware-version", "")))
        radios.append(
            {
                "iface": iface,
                "phy": _phy_for(iface),
                "pci": pci,
                "mac": _mac_for(iface),
                "operstate": _operstate(iface),
                "type": types.get(iface, "unknown"),
                "is_ax210": is_ax210,
                "lspci": ax_pci.get(pci, ""),
                "driver": eth.get("driver", ""),
                "kernel_module_version": eth.get("version", ""),
                "firmware": eth.get("firmware-version", ""),
            }
        )
    return radios


# --------------------------------------------------------------------------- #
# Role mapping (configs/nodes.yaml else iface-name heuristic)
# --------------------------------------------------------------------------- #
def _active_nodes(cfg: dict) -> dict:
    """Return the {nodeid: block} mapping for the active profile.

    Handles both the layered layout (profiles.<profile>.nodes) and a flat
    top-level `nodes` mapping. Degrades gracefully on either shape.
    """
    nodes = cfg.get("nodes")
    if isinstance(nodes, dict) and nodes:
        return nodes
    profiles = cfg.get("profiles") or {}
    profile = cfg.get("profile")
    if profile and profile in profiles:
        block = profiles[profile] or {}
        if isinstance(block.get("nodes"), dict):
            return block["nodes"]
    # else: first profile that has a nodes map.
    for block in profiles.values():
        if isinstance(block, dict) and isinstance(block.get("nodes"), dict):
            return block["nodes"]
    return {}


def load_node_config(node: str | None = None) -> dict:
    """Return the config block for `node` from configs/nodes.yaml.

    If node is None, resolve by matching this host's detected ifaces/MACs to a
    node entry; else fall back to the first node entry, else {}.
    """
    cfg = load_yaml(NODES_YAML)
    nodes = _active_nodes(cfg)
    if node and node in nodes:
        block = dict(nodes[node])
        block["_node_id"] = node
        return block
    # Resolve by MAC match against detected radios.
    my_macs = {r["mac"].lower() for r in detect_radios() if r["mac"]}
    for nid, block in nodes.items():
        radio_macs = {
            str(r.get("mac", "")).lower() for r in (block.get("radios") or [])
        }
        if my_macs & radio_macs:
            b = dict(block)
            b["_node_id"] = nid
            return b
    # Fallback: first node entry.
    for nid, block in nodes.items():
        b = dict(block)
        b["_node_id"] = nid
        return b
    return {}


def resolve_role_iface(role: str, node: str | None = None) -> dict:
    """Resolve the physical radio for a logical role ('csi'|'bfi'|...).

    Order: configs/nodes.yaml -> iface-name heuristic over detected AX210s.
    Returns a dict with at least {iface, phy, pci, mac, role, source}. If no
    radio can be resolved, iface is "" and source is "none".
    """
    block = load_node_config(node)
    node_id = block.get("_node_id") or node or hostname()
    for r in block.get("radios") or []:
        if str(r.get("role")) == role and r.get("iface"):
            return {
                "iface": r["iface"],
                "phy": r.get("phy", ""),
                "pci": r.get("pci", ""),
                "mac": r.get("mac", ""),
                "perspective": r.get("perspective"),
                "ap": r.get("ap", ""),
                "channel": r.get("channel"),
                "role": role,
                "node": node_id,
                "source": "nodes.yaml",
            }
    # Heuristic fallback: pick from detected AX210 radios. Convention from the
    # pilot node: lower PCI / phy1 (wlp1s0) -> csi, the other -> bfi.
    detected = [r for r in detect_radios() if r["is_ax210"]]
    detected.sort(key=lambda r: r["pci"])
    pick = None
    if detected:
        if role == "csi":
            pick = detected[0]
        else:  # bfi / bfi_recorder / anything else -> the second radio
            pick = detected[-1] if len(detected) > 1 else detected[0]
    if pick:
        return {
            "iface": pick["iface"],
            "phy": pick["phy"],
            "pci": pick["pci"],
            "mac": pick["mac"],
            "perspective": None,
            "ap": "",
            "channel": None,
            "role": role,
            "node": node_id,
            "source": "heuristic",
        }
    return {
        "iface": "",
        "phy": "",
        "pci": "",
        "mac": "",
        "perspective": None,
        "ap": "",
        "channel": None,
        "role": role,
        "node": node_id,
        "source": "none",
    }


# --------------------------------------------------------------------------- #
# Host / driver / clock environment
# --------------------------------------------------------------------------- #
def kernel_info() -> dict:
    u = os.uname()
    return {"sysname": u.sysname, "release": u.release, "machine": u.machine}


def clock_sources() -> dict:
    """Report availability/state of time-sync tooling (no root needed).

    chrony/ntp/ptp may be absent on a node; we report what's present rather than
    failing. The orchestrator owns the actual offset gate (configs/lab.yaml).
    """
    info: dict[str, Any] = {
        "chronyc": have_tool("chronyc"),
        "ntpq": have_tool("ntpq"),
        "ptp4l": have_tool("ptp4l"),
    }
    if info["chronyc"]:
        rc, out, _ = run(["chronyc", "tracking"], timeout=5)
        info["chrony_tracking_ok"] = rc == 0
        if rc == 0:
            for line in out.splitlines():
                if "System time" in line:
                    info["chrony_system_time"] = line.split(":", 1)[1].strip()
    return info


def tool_availability() -> dict:
    """Presence map of the tools the capture stack relies on."""
    return {
        t: have_tool(t)
        for t in (
            "iw",
            "tcpdump",
            "iperf3",
            "ethtool",
            "wpa_supplicant",
            "tshark",
            "chronyc",
            "ptp4l",
            "ssh",
        )
    }


# --------------------------------------------------------------------------- #
# Pidfile lifecycle (start writes, stop kills)
# --------------------------------------------------------------------------- #
def logs_dir_for(out_dir: Path | str) -> Path:
    """The trial logs/ dir where pidfiles + agent logs live."""
    return Path(out_dir) / "logs"


def pidfile_path(out_dir: Path | str, agent: str) -> Path:
    return logs_dir_for(out_dir) / f"{agent}.pid"


def write_pidfile(out_dir: Path | str, agent: str, pid: int, meta: dict | None = None) -> Path:
    """Persist {pid, cmd, ts} so a later `stop` can terminate the capture."""
    d = logs_dir_for(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pf = pidfile_path(out_dir, agent)
    payload = {"pid": int(pid), "ts_utc": now_utc()}
    if meta:
        payload.update(meta)
    pf.write_text(json.dumps(payload))
    return pf


def read_pidfile(out_dir: Path | str, agent: str) -> dict | None:
    pf = pidfile_path(out_dir, agent)
    if not pf.exists():
        return None
    try:
        return json.loads(pf.read_text())
    except Exception:
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    except Exception:
        return False


def stop_pidfile(out_dir: Path | str, agent: str, sig: int = signal.SIGTERM) -> dict:
    """Terminate the process named in the pidfile; remove the pidfile.

    Returns a result dict describing what happened (never raises).
    """
    info = read_pidfile(out_dir, agent)
    pf = pidfile_path(out_dir, agent)
    if not info:
        return {"stopped": False, "reason": "no pidfile", "pidfile": str(pf)}
    pid = int(info.get("pid", -1))
    result: dict[str, Any] = {"pidfile": str(pf), "pid": pid}
    if pid <= 0:
        result.update(stopped=False, reason="invalid pid")
    elif not pid_alive(pid):
        result.update(stopped=False, reason="process already exited")
    else:
        try:
            os.kill(pid, sig)
            result.update(stopped=True, signal=int(sig))
        except PermissionError:
            # Capture may have been launched as root (monitor mode); surface the
            # privileged kill rather than crashing.
            result.update(
                stopped=False,
                reason="permission denied",
                hint=privileged_hint(["kill", "-TERM", str(pid)]),
            )
        except Exception as exc:
            result.update(stopped=False, reason=str(exc))
    try:
        pf.unlink()
        result["pidfile_removed"] = True
    except Exception:
        result["pidfile_removed"] = False
    return result


# --------------------------------------------------------------------------- #
# Common argparse scaffold
# --------------------------------------------------------------------------- #
def add_common_args(parser) -> None:
    """Add the NODE-AGENT INVOCATION CONTRACT flags to an argparse parser."""
    parser.add_argument("action", choices=("detect", "start", "stop", "status", "health"))
    parser.add_argument("--participant")
    parser.add_argument("--style")
    parser.add_argument("--trial")
    parser.add_argument("--perspective", type=int)
    parser.add_argument("--out-dir", dest="out_dir",
                        help="trial directory (data/raw/.../trial=NNN). "
                             "Defaults to a path derived from contract if the "
                             "participant/style/trial are supplied.")
    parser.add_argument("--node", help="logical node id override (else resolved)")


def default_out_dir(args) -> Path | None:
    """Derive the canonical trial dir from contract if --out-dir is absent."""
    if getattr(args, "out_dir", None):
        return Path(args.out_dir)
    if args.participant and args.style and args.trial:
        try:
            from wallflower.contract import raw_trial_dir  # lazy: keep import optional

            lab = load_yaml(LAB_YAML)
            data_root = lab.get("data_root", "data")
            return raw_trial_dir(data_root, args.participant, args.style, args.trial)
        except Exception:
            return None
    return None
