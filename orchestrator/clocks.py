"""Clock-synchronisation gate for start-trial.

Synchronised timestamps across the 4 perspective nodes + recorder are required
to align CSI and BFI streams in post-processing (paper sec. 5.1). Before any
recording the orchestrator checks each node's clock offset against
lab.yaml clock.max_offset_ms and ABORTS if out of tolerance.

Source preference (configurable):
  ptp4l   -> read PTP master offset (if ptp4l/pmc present)
  chronyc -> 'chronyc tracking' "Last offset"
  ntpq    -> 'ntpq -pn' selected-peer offset
  ssh_date-> fallback: diff remote 'date +%s.%N' vs controller wall clock,
             correcting for measured round-trip (graceful degradation when no
             high-precision sync daemon is installed, as on this pilot box).

Returns a structured dict embedded into metadata.json (TrialMetadata.clock_sync)
and a boolean pass/fail. stdlib-only; tolerant of missing tools.
"""
from __future__ import annotations

import re
import time
from typing import Any, Sequence

from . import ssh


def _try_ptp(host: str, **kw: Any) -> float | None:
    """Best-effort PTP offset (ms). Returns None if ptp tooling absent."""
    rc, out, _err, error = ssh.run_remote_raw(
        host=host, cmd=["pmc", "-u", "-b", "0", "GET TIME_STATUS_NP"], timeout_s=10, **kw
    )
    if error or rc not in (0,) or not out:
        return None
    m = re.search(r"master_offset\s+(-?\d+)", out)
    if not m:
        return None
    return abs(int(m.group(1))) / 1e6  # ns -> ms


def _try_chrony(host: str, **kw: Any) -> float | None:
    rc, out, _err, error = ssh.run_remote_raw(
        host=host, cmd=["chronyc", "tracking"], timeout_s=10, **kw
    )
    if error or rc not in (0,) or not out:
        return None
    m = re.search(r"Last offset\s*:\s*([+-]?[0-9.eE-]+)\s*seconds", out)
    if not m:
        return None
    return abs(float(m.group(1))) * 1e3  # s -> ms


def _try_ntp(host: str, **kw: Any) -> float | None:
    rc, out, _err, error = ssh.run_remote_raw(
        host=host, cmd=["ntpq", "-pn"], timeout_s=10, **kw
    )
    if error or rc not in (0,) or not out:
        return None
    for line in out.splitlines():
        if line.startswith("*"):  # currently selected peer
            cols = line.split()
            if len(cols) >= 9:
                try:
                    return abs(float(cols[8]))  # offset column already in ms
                except ValueError:
                    return None
    return None


def _try_ssh_date(host: str, **kw: Any) -> float | None:
    """Fallback offset estimate (ms) via remote 'date' minus local wall clock.

    Corrects for measured round-trip time by assuming the remote read happened
    at the midpoint of the request. Local hosts trivially report ~0.
    """
    if ssh.is_local(host):
        return 0.0
    t0 = time.time()
    rc, out, _err, error = ssh.run_remote_raw(
        host=host, cmd=["date", "+%s.%N"], timeout_s=10, **kw
    )
    t1 = time.time()
    if error or rc not in (0,) or not out.strip():
        return None
    try:
        remote = float(out.strip().splitlines()[-1])
    except ValueError:
        return None
    midpoint = (t0 + t1) / 2.0
    return abs(remote - midpoint) * 1e3  # s -> ms


_SOURCES = {
    "ptp": _try_ptp,
    "chrony": _try_chrony,
    "ntp": _try_ntp,
    "ssh_date": _try_ssh_date,
}


def check_node_offset(
    host: str,
    prefer: Sequence[str],
    *,
    ssh_user: str | None = None,
    ssh_options: Sequence[str] | None = None,
    connect_timeout_s: int = 8,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Estimate one node's clock offset (ms), trying sources in `prefer` order."""
    if dry_run:
        return {"host": host, "source": "dry_run", "offset_ms": 0.0, "degraded": False, "ok": True}

    kw = dict(ssh_user=ssh_user, ssh_options=ssh_options, connect_timeout_s=connect_timeout_s)
    high_precision = {"ptp", "chrony", "ntp"}
    for name in prefer:
        fn = _SOURCES.get(name)
        if fn is None:
            continue
        offset = fn(host, **kw)
        if offset is not None:
            return {
                "host": host,
                "source": name,
                "offset_ms": round(offset, 4),
                "degraded": name not in high_precision,
            }
    return {"host": host, "source": None, "offset_ms": None, "degraded": True,
            "error": "no clock source produced an offset"}


def check_sync(
    hosts: dict[str, str],
    *,
    max_offset_ms: float,
    prefer: Sequence[str],
    allow_degraded: bool = True,
    ssh_user: str | None = None,
    ssh_options: Sequence[str] | None = None,
    connect_timeout_s: int = 8,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Check clock sync across a set of {node: host} entries.

    Returns a structured dict suitable for TrialMetadata.clock_sync:
      { ok, max_offset_ms, tolerance_ms, degraded, nodes: {node: {...}},
        warnings: [...], errors: [...] }
    `ok` is False if any reachable node exceeds tolerance, or if a node has no
    offset at all. Degraded (low-precision ssh_date) sources are a warning when
    allow_degraded, else an error.
    """
    report: dict[str, Any] = {
        "ok": True,
        "tolerance_ms": max_offset_ms,
        "max_offset_ms": 0.0,
        "degraded": False,
        "nodes": {},
        "warnings": [],
        "errors": [],
    }
    seen_offsets: list[float] = []

    # De-duplicate by host so we don't probe the same machine repeatedly.
    host_to_nodes: dict[str, list[str]] = {}
    for node, host in hosts.items():
        host_to_nodes.setdefault(host, []).append(node)

    host_results: dict[str, dict[str, Any]] = {}
    for host in host_to_nodes:
        host_results[host] = check_node_offset(
            host, prefer,
            ssh_user=ssh_user, ssh_options=ssh_options,
            connect_timeout_s=connect_timeout_s, dry_run=dry_run,
        )

    for node, host in hosts.items():
        r = dict(host_results[host])  # copy per node
        report["nodes"][node] = r
        off = r.get("offset_ms")
        if off is None:
            report["ok"] = False
            report["errors"].append(f"{node} ({host}): no clock offset available")
            continue
        seen_offsets.append(off)
        if r.get("degraded"):
            report["degraded"] = True
            msg = f"{node} ({host}): degraded clock source '{r.get('source')}' offset={off}ms"
            if allow_degraded:
                report["warnings"].append(msg)
            else:
                report["ok"] = False
                report["errors"].append(msg + " (high-precision sync required)")
        if off > max_offset_ms:
            report["ok"] = False
            report["errors"].append(
                f"{node} ({host}): offset {off}ms exceeds tolerance {max_offset_ms}ms"
            )

    report["max_offset_ms"] = round(max(seen_offsets), 4) if seen_offsets else None
    return report
