"""Aggregate node health report for wallflower.

Combines radio detection, role mapping, driver/kernel/firmware, clock-sync tool
state and capture-tool availability into ONE structured JSON object. Usable both
standalone (operator: `python3 -m nodes.healthcheck`) and as an import by the
orchestrator (`from nodes.healthcheck import gather_health`).

Imports wallflower.contract so the report's expected RF facts (channels, band, width,
modality feature dims, nominal rates) come from the single source of truth and
cannot drift.

STDLIB-ONLY (degrades gracefully on a bare node).
"""
from __future__ import annotations

import argparse

from wallflower.contract import (
    AP_CHANNELS,
    BAND_GHZ,
    FEATURE_DIMS,
    NOMINAL_RATE_HZ,
    RADIO_ROLES,
    WIDTH_MHZ,
)

from . import common

AGENT = "healthcheck"

# Tools the capture stack relies on; value notes whether absence is fatal for a
# real capture (the brief lists several as known-missing on the pilot node).
REQUIRED_TOOLS = {
    "iw": "required (channel/monitor control)",
    "tcpdump": "required (BFI pcap capture)",
    "ethtool": "optional (driver/firmware introspection)",
    "wpa_supplicant": "optional (secured AP-BFI association)",
    "iperf3": "required for traffic_agent (known missing on pilot)",
    "tshark": "optional (offline pcap inspection; known missing)",
    "chronyc": "optional (clock-sync source)",
    "ptp4l": "optional (high-precision clock-sync; known missing)",
    "ssh": "required on controller for remote nodes",
}


def _radio_role_summary(node: str | None) -> dict:
    """Resolve each logical role to a physical radio and flag conflicts."""
    summary: dict[str, dict] = {}
    seen_ifaces: dict[str, str] = {}
    for role in RADIO_ROLES:
        r = common.resolve_role_iface(role, node)
        summary[role] = r
        if r["iface"]:
            if r["iface"] in seen_ifaces and seen_ifaces[r["iface"]] != role:
                r["conflict"] = (f"iface {r['iface']} also mapped to role "
                                 f"{seen_ifaces[r['iface']]}")
            seen_ifaces[r["iface"]] = role
    return summary


def gather_health(node: str | None = None) -> dict:
    """Build (do NOT print) the full health dict. Importable by orchestrator."""
    radios = common.detect_radios()
    ax210 = [r for r in radios if r["is_ax210"]]
    roles = _radio_role_summary(node)
    tools = common.tool_availability()

    # Health gate: at least 2 AX210s detected, csi+bfi roles resolved, iw+tcpdump
    # present, driver is iwlwifi on the AX210s.
    drivers_ok = all(r.get("driver") == "iwlwifi" for r in ax210) if ax210 else False
    roles_ok = bool(roles.get("csi", {}).get("iface")) and \
        bool(roles.get("bfi", {}).get("iface"))
    tools_ok = tools.get("iw", False) and tools.get("tcpdump", False)
    ok = len(ax210) >= 2 and roles_ok and tools_ok and drivers_ok

    node_id = roles.get("csi", {}).get("node") or node or common.hostname()
    return {
        "node": node_id,
        "ok": ok,
        "radios": radios,
        "ax210_count": len(ax210),
        "roles": roles,
        "kernel": common.kernel_info(),
        "clock": common.clock_sources(),
        "tools": tools,
        "tool_requirements": REQUIRED_TOOLS,
        "missing_required_tools": [
            t for t, note in REQUIRED_TOOLS.items()
            if note.startswith("required") and not tools.get(t, False)
        ],
        "is_root": common.is_root(),
        "rf_expected": {
            "ap_channels": dict(AP_CHANNELS),
            "band_ghz": BAND_GHZ,
            "width_mhz": WIDTH_MHZ,
        },
        "modality_feature_dims": dict(FEATURE_DIMS),
        "nominal_rate_hz": dict(NOMINAL_RATE_HZ),
        "warnings": _warnings(ax210, roles, tools, drivers_ok),
    }


def _warnings(ax210: list, roles: dict, tools: dict, drivers_ok: bool) -> list[str]:
    w: list[str] = []
    if len(ax210) < 2:
        w.append(f"expected 2 AX210 radios, detected {len(ax210)}")
    if not drivers_ok:
        w.append("not all AX210 radios report the iwlwifi driver")
    for role in RADIO_ROLES:
        if not roles.get(role, {}).get("iface"):
            w.append(f"no radio resolved for role {role!r}")
        elif "conflict" in roles[role]:
            w.append(roles[role]["conflict"])
    if not tools.get("iperf3"):
        w.append("iperf3 missing: run `sudo apt-get install -y iperf3` "
                 "(traffic_agent degrades without it)")
    if not tools.get("chronyc") and not tools.get("ptp4l"):
        w.append("no chrony/ptp time source: clock-sync gate will rely on "
                 "ssh-date fallback")
    if not common.is_root():
        w.append("not root: monitor-mode/channel/capture commands must be run "
                 "by the operator with sudo (agents print the exact commands)")
    return w


def health(node: str | None) -> dict:
    rep = gather_health(node)
    # Emit as the standard structured log; nest the detail under 'report'.
    return common.emit(AGENT, "health", rep["ok"], node=rep["node"], report=rep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m nodes.healthcheck")
    # healthcheck only meaningfully supports 'health'/'detect'; accept the full
    # action set for contract uniformity (all map to the health report).
    parser.add_argument("action", nargs="?", default="health",
                        choices=("detect", "start", "stop", "status", "health"))
    parser.add_argument("--node", help="logical node id override")
    # Tolerate the common trial flags so the orchestrator can call this agent
    # with the same argument shape as the others.
    parser.add_argument("--participant")
    parser.add_argument("--style")
    parser.add_argument("--trial")
    parser.add_argument("--perspective", type=int)
    parser.add_argument("--out-dir", dest="out_dir")
    args = parser.parse_args(argv)
    result = health(args.node)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
