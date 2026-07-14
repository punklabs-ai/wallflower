"""`wallflower` orchestrator CLI — runs an experiment session from the controller.

Reproduction of BFId (Todt, Morsbach, Strufe, CCS '25). Records synchronised
CSI + BFI walking traces from a 4-perspective lab and writes the canonical raw
layout + per-trial metadata.json.

Subcommands:
  wallflower init-session     --participant P001 --style normal
  wallflower start-trial      --participant P001 --style normal --trial 001
  wallflower stop-trial       --participant P001 --style normal --trial 001
  wallflower validate-session --participant P001

Hosts come only from the nodes.yaml inventory.

Node coordination follows the NODE-AGENT INVOCATION CONTRACT:
  python3 -m nodes.<agent> <action> --participant ... --style ... --trial ...
Clock sync is enforced before recording (clocks.py) against lab.yaml.

stdlib + pyyaml only; in --profile pilot localhost runs agents directly (no ssh).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from wallflower import contract

from . import clocks, metadata as meta_mod, session as session_mod, ssh

# Default config locations (relative to repo root / cwd).
DEFAULT_LAB = "configs/lab.yaml"
DEFAULT_NODES = "configs/nodes.yaml"

# Order in which capture/traffic agents are started (and reverse-stopped).
# Recorder + CSI capture first (so they are armed), then traffic + bfi clients
# that actually generate the soundings.
AGENT_ROLES = ("bfi_recorder", "csi", "csi_traffic", "bfi_client")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(obj: dict[str, Any]) -> None:
    """Print ONE structured JSON object (matches the node-agent convention)."""
    print(json.dumps(obj, indent=2, sort_keys=False, default=str))


def _base_log(action: str, **kw: Any) -> dict[str, Any]:
    log = {
        "agent": "orchestrator",
        "action": action,
        "ok": True,
        "node": "controller",
        "ts_utc": _now_utc_iso(),
    }
    log.update(kw)
    return log


# --------------------------------------------------------------------------- #
# Agent fan-out helpers
# --------------------------------------------------------------------------- #
def _ssh_kwargs(cfg: session_mod.LabConfig, node: session_mod.NodeInfo) -> dict[str, Any]:
    return {
        "ssh_user": node.user or cfg.ssh_user or None,
        "ssh_options": cfg.ssh_options,
        "connect_timeout_s": cfg.ssh_connect_timeout_s,
    }


def _plan_recording_agents(
    cfg: session_mod.LabConfig, perspectives: list[int], out_dir: str,
    participant: str, style: str, trial: str,
) -> list[dict[str, Any]]:
    """Return the ordered list of agent invocations for a trial (start/stop)."""
    plan: list[dict[str, Any]] = []

    # 1. Central passive BFI recorder (single pcapng for all perspectives).
    rec = cfg.recorder_node()
    if rec:
        plan.append({
            "agent": "bfi_recorder_agent", "node": rec.node, "host": rec.host,
            "perspective": None, "ssh": _ssh_kwargs(cfg, rec),
            "role": "bfi_recorder",
        })

    # 2. CSI capture per perspective.
    for p in perspectives:
        node = cfg.csi_node_for(p)
        if not node:
            continue
        plan.append({
            "agent": "csi_agent", "node": node.node, "host": node.host,
            "perspective": p, "ssh": _ssh_kwargs(cfg, node),
            "role": "csi",
        })

    # 3. Traffic generators (drive BFI sounding + CSI).
    for node in cfg.nodes_with_role("csi_traffic"):
        plan.append({
            "agent": "traffic_agent", "node": node.node, "host": node.host,
            "perspective": None, "ssh": _ssh_kwargs(cfg, node),
            "role": "csi_traffic",
        })

    # 4. BFI clients (associate to AP-BFI to elicit sounding) per perspective.
    for p in perspectives:
        node = cfg.csi_node_for(p)
        if not node or "bfi" not in node.roles:
            # bfi client may be the same physical node in pilot
            node = cfg.csi_node_for(p)
        if node and ("bfi" in node.roles or cfg.radio_for_role(node, "bfi")):
            plan.append({
                "agent": "bfi_client_agent", "node": node.node, "host": node.host,
                "perspective": p, "ssh": _ssh_kwargs(cfg, node),
                "role": "bfi_client",
            })
    return plan


def _invoke_one(item: dict[str, Any], action: str, *, participant: str,
                style: str, trial: str, out_dir: str, dry_run: bool,
                extra: Sequence[str] | dict[str, Sequence[str]] | None) -> ssh.AgentResult:
    """Invoke a single agent. ssh.run_agent never raises, but we also guard here
    so one agent blowing up can never abort a sibling's launch."""
    agent_extra: Sequence[str] | None
    if isinstance(extra, dict):
        agent_extra = [*extra.get("*", ()), *extra.get(item["agent"], ())]
    else:
        agent_extra = extra
    try:
        return ssh.run_agent(
            agent=item["agent"], action=action, node=item["node"], host=item["host"],
            participant=participant, style=style, trial=trial,
            perspective=item.get("perspective"), out_dir=out_dir,
            extra=agent_extra, dry_run=dry_run, **item["ssh"],
        )
    except Exception as exc:  # noqa: BLE001 — isolation: never let one agent abort others
        return ssh.AgentResult(
            agent=item["agent"], action=action, node=item["node"],
            host=item["host"], ok=False, perspective=item.get("perspective"),
            error=f"orchestrator invoke error: {exc}",
        )


def _run_plan(plan: list[dict[str, Any]], action: str, *, participant: str,
              style: str, trial: str, out_dir: str, dry_run: bool,
              reverse: bool = False,
              extra: Sequence[str] | dict[str, Sequence[str]] | None = None) -> list[ssh.AgentResult]:
    results: list[ssh.AgentResult] = []
    items = list(reversed(plan)) if reverse else plan
    for item in items:
        results.append(_invoke_one(
            item, action, participant=participant, style=style, trial=trial,
            out_dir=out_dir, dry_run=dry_run, extra=extra))
    return results


def _run_plan_concurrent(plan: list[dict[str, Any]], action: str, *,
                         participant: str, style: str, trial: str, out_dir: str,
                         dry_run: bool,
                         extra: Sequence[str] | dict[str, Sequence[str]] | None = None
                         ) -> list[ssh.AgentResult]:
    """Fan out the whole plan CONCURRENTLY so CSI capture and the BFI chain
    (recorder + traffic + client) are armed together for the SAME trial, exactly
    as BFId recorded both modalities in one session.

    Failure isolation: each invocation runs in its own thread and its exceptions
    are caught, so a CSI failure cannot prevent the BFI capture from launching
    (and vice-versa). Results are returned in plan order regardless of finish
    order. Each agent itself Popen-detaches the real long-running capture, so
    the threads return promptly.
    """
    if not plan:
        return []
    results: list[ssh.AgentResult | None] = [None] * len(plan)
    with ThreadPoolExecutor(max_workers=len(plan)) as ex:
        futs = {
            ex.submit(
                _invoke_one, item, action, participant=participant, style=style,
                trial=trial, out_dir=out_dir, dry_run=dry_run, extra=extra): i
            for i, item in enumerate(plan)
        }
        for fut, i in futs.items():
            results[i] = fut.result()
    return [r for r in results if r is not None]


# --------------------------------------------------------------------------- #
# Subcommand: init-session
# --------------------------------------------------------------------------- #
def cmd_init_session(args: argparse.Namespace) -> int:
    cfg = session_mod.load_config(args.config, args.nodes, args.profile)
    log = _base_log("init-session", participant=args.participant, profile=cfg.profile)
    try:
        contract.validate_participant(args.participant)
        if args.style:
            contract.validate_style(args.style)
    except ValueError as exc:
        log.update(ok=False, error=str(exc))
        _emit(log)
        return 2

    styles = [args.style] if args.style else None
    sess = session_mod.init_session(
        cfg.data_root, args.participant,
        style=args.style, styles=styles,
        operator=args.operator or "",
        dry_run=args.dry_run,
    )
    log["session"] = sess
    log["dry_run"] = args.dry_run
    if not args.dry_run:
        log["session_json"] = str(session_mod.session_json_path(cfg.data_root, args.participant))
    _emit(log)
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: start-trial
# --------------------------------------------------------------------------- #
def cmd_start_trial(args: argparse.Namespace) -> int:
    cfg = session_mod.load_config(args.config, args.nodes, args.profile)
    perspectives = _resolve_perspectives(args, cfg)
    direction = args.direction or cfg.lab.get("default_direction", contract.DIRECTIONS[0])
    log = _base_log("start-trial", participant=args.participant, style=args.style,
                    trial=args.trial, profile=cfg.profile, perspectives=perspectives,
                    direction=direction, dry_run=args.dry_run)

    # --- validate identifiers ---
    try:
        contract.validate_participant(args.participant)
        contract.validate_style(args.style)
        contract.validate_trial(args.trial)
        if direction not in contract.DIRECTIONS:
            raise ValueError(f"direction {direction!r} not in {contract.DIRECTIONS}")
    except ValueError as exc:
        log.update(ok=False, error=str(exc))
        _emit(log)
        return 2

    # --- CLOCK SYNC GATE ---
    clk_cfg = cfg.lab.get("clock", {}) or {}
    hosts = cfg.all_record_hosts(perspectives)
    clock_sync = clocks.check_sync(
        hosts,
        max_offset_ms=float(clk_cfg.get("max_offset_ms", 50.0)),
        prefer=clk_cfg.get("prefer", ["ptp", "chrony", "ntp", "ssh_date"]),
        allow_degraded=bool(clk_cfg.get("allow_degraded", True)),
        ssh_user=cfg.ssh_user or None,
        ssh_options=cfg.ssh_options,
        connect_timeout_s=cfg.ssh_connect_timeout_s,
        dry_run=args.dry_run,
    )
    log["clock_sync"] = clock_sync
    if not clock_sync["ok"] and not args.skip_clock_check:
        log.update(
            ok=False,
            error="clock_out_of_sync",
            message=(
                "ABORTING start-trial: clock sync check failed "
                f"(tolerance {clock_sync['tolerance_ms']}ms). "
                "Fix sync (chrony/ptp) or, only for a dry run, pass "
                "--skip-clock-check. Errors: " + "; ".join(clock_sync["errors"])
            ),
        )
        _emit(log)
        return 4

    out_dir = str(contract.raw_trial_dir(cfg.data_root, args.participant, args.style, args.trial))

    # --- health probe (best-effort) to enrich radios ---
    health = _gather_health(cfg, perspectives, dry_run=args.dry_run)

    # --- shared capture-start epoch: ONE wall clock for BOTH modalities so the
    #     concurrently-recorded CSI and BFI streams can be time-aligned. Stamped
    #     immediately before the concurrent fan-out below. ---
    capture_start_epoch = time.time()

    # --- write metadata.json BEFORE capture so partial trials are documented ---
    meta = meta_mod.build_trial_metadata(
        cfg, participant=args.participant, style=args.style, trial=args.trial,
        direction=direction, perspectives=perspectives, clock_sync=clock_sync,
        health=health, notes=args.notes or "",
        capture_start_epoch=capture_start_epoch,
    )
    if not args.dry_run:
        meta_path = meta_mod.write_trial_metadata(cfg.data_root, meta)
        log["metadata_json"] = str(meta_path)
    else:
        log["metadata_json"] = str(contract.raw_trial_dir(
            cfg.data_root, args.participant, args.style, args.trial) / contract.metadata_name())
    log["metadata"] = meta.to_dict()

    # --- fan out start to agents CONCURRENTLY ---
    # CSI capture (csi_agent on the csi-role radio) and the BFI chain
    # (bfi_recorder + traffic + bfi_client) are launched together for ONE trial,
    # not sequentially. A failure in one modality does NOT abort the other's
    # capture (per-thread failure isolation in _run_plan_concurrent).
    plan = _plan_recording_agents(cfg, perspectives, out_dir,
                                  args.participant, args.style, args.trial)
    common_extra: list[str] = []
    traffic_extra: list[str] = []
    if getattr(args, "bfi_server", None):
        traffic_extra.extend(["--bfi-server", args.bfi_server])
    if getattr(args, "csi_server", None):
        traffic_extra.extend(["--csi-server", args.csi_server])
    if getattr(args, "allow_traffic_fallback", False):
        traffic_extra.append("--allow-fallback")
    extra = {"*": common_extra, "traffic_agent": traffic_extra}
    log["capture_start_epoch"] = capture_start_epoch
    log["concurrent"] = True
    results = _run_plan_concurrent(plan, "start", participant=args.participant,
                                   style=args.style, trial=args.trial,
                                   out_dir=out_dir, dry_run=args.dry_run,
                                   extra=extra)
    log["agents"] = [r.to_dict() for r in results]
    failed = [r for r in results if not r.ok]
    if failed:
        # Both modalities were still attempted concurrently; report which legs
        # failed without implying the others did not capture.
        ok_agents = [r for r in results if r.ok]
        log.update(
            ok=False,
            error="agent_start_failed",
            message=("Some capture agents failed to start (the others were still "
                     "launched concurrently and may be capturing). Run "
                     f"`wallflower stop-trial --participant {args.participant} "
                     f"--style {args.style} --trial {args.trial}` to clean up. "
                     "Failed: " + ", ".join(f"{r.node}:{r.agent}" for r in failed)
                     + ("; started: " + ", ".join(f"{r.node}:{r.agent}"
                                                   for r in ok_agents)
                        if ok_agents else "")),
        )
        _emit(log)
        return 5

    log["message"] = (
        "Recording started. Walk the protocol leg, then run stop-trial."
        if not args.dry_run else "DRY RUN: no hardware touched."
    )
    _emit(log)
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: stop-trial
# --------------------------------------------------------------------------- #
def cmd_stop_trial(args: argparse.Namespace) -> int:
    cfg = session_mod.load_config(args.config, args.nodes, args.profile)
    perspectives = _resolve_perspectives(args, cfg)
    log = _base_log("stop-trial", participant=args.participant, style=args.style,
                    trial=args.trial, profile=cfg.profile, perspectives=perspectives,
                    dry_run=args.dry_run)
    try:
        contract.validate_participant(args.participant)
        contract.validate_style(args.style)
        contract.validate_trial(args.trial)
    except ValueError as exc:
        log.update(ok=False, error=str(exc))
        _emit(log)
        return 2

    out_dir = str(contract.raw_trial_dir(cfg.data_root, args.participant, args.style, args.trial))
    plan = _plan_recording_agents(cfg, perspectives, out_dir,
                                  args.participant, args.style, args.trial)
    # Stop in reverse start order: kill traffic/clients first, then captures.
    extra = None
    results = _run_plan(plan, "stop", participant=args.participant, style=args.style,
                        trial=args.trial, out_dir=out_dir, dry_run=args.dry_run,
                        reverse=True, extra=extra)
    log["agents"] = [r.to_dict() for r in results]
    failed = [r for r in results if not r.ok]
    if failed:
        log.update(
            ok=False, error="agent_stop_failed",
            message=("Some agents failed to stop cleanly (check pidfiles on nodes): "
                     + ", ".join(f"{r.node}:{r.agent}" for r in failed)),
        )
        _emit(log)
        return 5
    log["message"] = "Recording stopped." if not args.dry_run else "DRY RUN: no hardware touched."
    _emit(log)
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: validate-session
# --------------------------------------------------------------------------- #
def cmd_validate_session(args: argparse.Namespace) -> int:
    cfg = session_mod.load_config(args.config, args.nodes, args.profile)
    contract.validate_participant(args.participant)
    report = validate_session(cfg, args.participant)
    _emit(report)
    return 0 if report["ok"] else 1


def validate_session(cfg: session_mod.LabConfig, participant: str) -> dict[str, Any]:
    """Walk data/raw for a participant and produce a structured ok/warn/error report."""
    data_root = cfg.data_root
    pdir = session_mod.participant_dir(data_root, participant)
    report: dict[str, Any] = {
        "agent": "orchestrator",
        "action": "validate-session",
        "ok": True,
        "node": "controller",
        "ts_utc": _now_utc_iso(),
        "participant": participant,
        "participant_dir": str(pdir),
        "styles": {},
        "warnings": [],
        "errors": [],
        "trial_count": 0,
    }

    if not pdir.exists():
        report["ok"] = False
        report["errors"].append(f"participant dir does not exist: {pdir}")
        return report

    expected_perspectives = list(cfg.lab.get("perspectives", list(contract.PERSPECTIVES)))

    for style in contract.WALKING_STYLES:
        sdir = pdir / f"style={style}"
        if not sdir.exists():
            continue
        trial_dirs = sorted(d for d in sdir.iterdir()
                            if d.is_dir() and d.name.startswith("trial="))
        style_report: dict[str, Any] = {
            "trials_found": len(trial_dirs),
            "expected_repeats": contract.STYLE_REPEATS[style],
            "trials": {},
        }
        if len(trial_dirs) != contract.STYLE_REPEATS[style]:
            report["warnings"].append(
                f"style {style}: {len(trial_dirs)} trials found, "
                f"expected {contract.STYLE_REPEATS[style]} (STYLE_REPEATS)")

        for td in trial_dirs:
            report["trial_count"] += 1
            trial_id = td.name.split("=", 1)[1]
            tr = _validate_trial_dir(td, expected_perspectives)
            style_report["trials"][trial_id] = tr
            for e in tr["errors"]:
                report["errors"].append(f"{style}/{trial_id}: {e}")
            for w in tr["warnings"]:
                report["warnings"].append(f"{style}/{trial_id}: {w}")
        report["styles"][style] = style_report

    report["ok"] = not report["errors"]
    return report


def _validate_trial_dir(td: Path, expected_perspectives: list[int]) -> dict[str, Any]:
    tr: dict[str, Any] = {"dir": str(td), "errors": [], "warnings": [], "files": {}}

    # metadata.json
    meta_path = td / contract.metadata_name()
    meta: dict[str, Any] | None = None
    if not meta_path.exists():
        tr["errors"].append("missing metadata.json")
    elif meta_path.stat().st_size == 0:
        tr["errors"].append("empty metadata.json")
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            tr["errors"].append(f"metadata.json not valid JSON: {exc}")
        if meta is not None:
            for err in meta_mod.validate_metadata_dict(meta):
                tr["errors"].append(err)

    # active perspectives: from metadata if present else lab.yaml
    active = (meta.get("perspectives") if meta else None) or expected_perspectives

    # csi_p*.raw — one per active perspective
    for p in active:
        name = contract.csi_raw_name(p)
        f = td / name
        if not f.exists():
            tr["errors"].append(f"missing {name}")
        elif f.stat().st_size == 0:
            tr["errors"].append(f"empty {name}")
        else:
            tr["files"][name] = f.stat().st_size

    # bfi_recorder.pcapng
    bfi = td / contract.bfi_recorder_name()
    if not bfi.exists():
        tr["errors"].append(f"missing {contract.bfi_recorder_name()}")
    elif bfi.stat().st_size == 0:
        tr["errors"].append(f"empty {contract.bfi_recorder_name()}")
    else:
        tr["files"][contract.bfi_recorder_name()] = bfi.stat().st_size

    # logs/
    logs = td / "logs"
    if not logs.exists() or not logs.is_dir():
        tr["warnings"].append("missing logs/ directory")

    return tr


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _resolve_perspectives(args: argparse.Namespace,
                          cfg: session_mod.LabConfig) -> list[int]:
    if getattr(args, "perspectives", None):
        out = []
        for tok in str(args.perspectives).replace(",", " ").split():
            out.append(contract.validate_perspective(int(tok)))
        return out
    return [contract.validate_perspective(int(p))
            for p in cfg.lab.get("perspectives", list(contract.PERSPECTIVES))]


def _gather_health(cfg: session_mod.LabConfig, perspectives: list[int],
                   *, dry_run: bool) -> dict[str, dict[str, Any]]:
    """Best-effort 'health' probe of each recording node (never fatal)."""
    health: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    plan = _plan_recording_agents(cfg, perspectives, "", "P000", "normal", "000")
    for item in plan:
        node = item["node"]
        if node in seen:
            continue
        seen.add(node)
        r = ssh.run_agent(
            agent=item["agent"], action="health", node=node, host=item["host"],
            dry_run=dry_run, **item["ssh"],
        )
        if r.json:
            health[node] = r.json
    return health


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=DEFAULT_LAB, help="lab config yaml (default %(default)s)")
    p.add_argument("--nodes", default=DEFAULT_NODES, help="nodes inventory yaml (default %(default)s)")
    p.add_argument("--profile", default=None, help="node profile (default: nodes.yaml 'profile')")
    p.add_argument("--dry-run", action="store_true",
                   help="print planned actions without touching hardware")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wallflower",
        description="wallflower orchestrator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init-session
    p_init = sub.add_parser("init-session", help="scaffold participant dir + session")
    _add_common(p_init)
    p_init.add_argument("--participant", required=True)
    p_init.add_argument("--style", default=None,
                        help="optional style to scaffold (default: all WALKING_STYLES)")
    p_init.add_argument("--operator", default="", help="operator name/id for the record")
    p_init.set_defaults(func=cmd_init_session)

    # start-trial
    p_start = sub.add_parser("start-trial", help="start synchronised CSI+BFI capture")
    _add_common(p_start)
    p_start.add_argument("--participant", required=True)
    p_start.add_argument("--style", required=True)
    p_start.add_argument("--trial", required=True)
    p_start.add_argument("--direction", default=None, choices=list(contract.DIRECTIONS),
                         help="back-and-forth leg (default: lab.yaml default_direction)")
    p_start.add_argument("--perspectives", default=None,
                         help="comma/space list, e.g. '1,2,3,4' (default: lab.yaml)")
    p_start.add_argument("--notes", default="", help="free-text notes for metadata")
    p_start.add_argument("--skip-clock-check", action="store_true",
                         help="bypass the clock-sync gate (dev/dry-run only)")
    p_start.add_argument("--bfi-server", dest="bfi_server", default=None,
                         help="iperf3 server for the BFI sounding TCP flow")
    p_start.add_argument("--csi-server", dest="csi_server", default=None,
                         help="iperf3 server for the CSI keepalive UDP flow")
    p_start.add_argument("--allow-traffic-fallback", dest="allow_traffic_fallback",
                         action="store_true",
                         help="forward --allow-fallback to traffic_agent when iperf3 "
                              "is unavailable")
    p_start.set_defaults(func=cmd_start_trial)

    # stop-trial
    p_stop = sub.add_parser("stop-trial", help="stop capture + traffic agents")
    _add_common(p_stop)
    p_stop.add_argument("--participant", required=True)
    p_stop.add_argument("--style", required=True)
    p_stop.add_argument("--trial", required=True)
    p_stop.add_argument("--perspectives", default=None,
                        help="comma/space list (default: lab.yaml)")
    p_stop.set_defaults(func=cmd_stop_trial)

    # validate-session
    p_val = sub.add_parser("validate-session", help="check raw layout for a participant")
    _add_common(p_val)
    p_val.add_argument("--participant", required=True)
    p_val.set_defaults(func=cmd_validate_session)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        _emit(_base_log(getattr(args, "command", "?"), ok=False, error=f"config_not_found: {exc}"))
        return 2
    except ValueError as exc:
        _emit(_base_log(getattr(args, "command", "?"), ok=False, error=str(exc)))
        return 2


if __name__ == "__main__":
    sys.exit(main())
