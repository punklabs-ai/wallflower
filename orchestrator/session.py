"""Session model + config loading for the orchestrator.

Loads lab.yaml and nodes.yaml, resolves the active --profile into a concrete
node inventory, and provides the participant/session scaffold helpers used by
`wallflower init-session`.

nodes.yaml layout (authoritative):
  profile: pilot                # default active profile name
  topology.full: {...}          # logical reference for full 4-perspective rollout
  profiles.<name>:
    nodes: {node: {host,user,roles,radios:[...]}}
    controller_node / bfi_recorder_node / trainer_node: <node>
    perspective_nodes: {1: node, ...}
  ssh: {user, connect_timeout_s, options:[...]}

stdlib + pyyaml only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from wallflower import contract


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config {p} did not parse to a mapping")
    return data


@dataclass
class NodeInfo:
    """One resolved node from the active profile."""
    node: str
    host: str
    user: str = ""
    roles: list[str] = field(default_factory=list)
    radios: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LabConfig:
    """Merged, resolved configuration for a session."""
    lab: dict[str, Any]
    nodes_raw: dict[str, Any]
    profile: str
    data_root: Path
    nodes: dict[str, NodeInfo]
    perspective_nodes: dict[int, str]
    bfi_recorder_node: str | None
    ssh_user: str
    ssh_options: list[str]
    ssh_connect_timeout_s: int

    # ---- role / perspective resolution -----------------------------------
    def node(self, name: str) -> NodeInfo | None:
        return self.nodes.get(name)

    def csi_node_for(self, perspective: int) -> NodeInfo | None:
        name = self.perspective_nodes.get(perspective)
        return self.nodes.get(name) if name else None

    def recorder_node(self) -> NodeInfo | None:
        if self.bfi_recorder_node:
            return self.nodes.get(self.bfi_recorder_node)
        for n in self.nodes.values():
            if "bfi_recorder" in n.roles:
                return n
        return None

    def nodes_with_role(self, role: str) -> list[NodeInfo]:
        return [n for n in self.nodes.values() if role in n.roles]

    def radio_for_role(self, node: NodeInfo, role: str,
                       perspective: int | None = None) -> dict[str, Any] | None:
        for r in node.radios:
            if r.get("role") == role:
                if perspective is None or r.get("perspective") in (None, perspective):
                    return r
        return None

    def all_record_hosts(self, perspectives: list[int]) -> dict[str, str]:
        """{node: host} for every node taking part in recording (clock check)."""
        out: dict[str, str] = {}
        for p in perspectives:
            n = self.csi_node_for(p)
            if n:
                out[n.node] = n.host
        rec = self.recorder_node()
        if rec:
            out[rec.node] = rec.host
        return out


def load_config(lab_path: Path | str, nodes_path: Path | str,
                profile: str | None = None) -> LabConfig:
    """Load + merge lab.yaml and nodes.yaml and resolve the active profile."""
    lab = load_yaml(lab_path)
    nodes_raw = load_yaml(nodes_path)

    active = profile or nodes_raw.get("profile") or "pilot"
    profiles = nodes_raw.get("profiles", {})
    if active not in profiles:
        raise ValueError(
            f"profile {active!r} not found in {nodes_path} (have: {list(profiles)})"
        )
    prof = profiles[active]

    nodes: dict[str, NodeInfo] = {}
    for name, spec in (prof.get("nodes") or {}).items():
        nodes[name] = NodeInfo(
            node=name,
            host=spec.get("host", "localhost"),
            user=spec.get("user", ""),
            roles=list(spec.get("roles", [])),
            radios=list(spec.get("radios", [])),
        )

    persp_nodes: dict[int, str] = {}
    for k, v in (prof.get("perspective_nodes") or {}).items():
        persp_nodes[int(k)] = v

    ssh_cfg = nodes_raw.get("ssh", {}) or {}
    data_root = Path(lab.get("data_root", "data"))

    return LabConfig(
        lab=lab,
        nodes_raw=nodes_raw,
        profile=active,
        data_root=data_root,
        nodes=nodes,
        perspective_nodes=persp_nodes,
        bfi_recorder_node=prof.get("bfi_recorder_node"),
        ssh_user=ssh_cfg.get("user", ""),
        ssh_options=list(ssh_cfg.get("options", [])),
        ssh_connect_timeout_s=int(ssh_cfg.get("connect_timeout_s", 8)),
    )


# --------------------------------------------------------------------------- #
# Session scaffold (init-session)
# --------------------------------------------------------------------------- #
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def participant_dir(data_root: Path | str, participant: str) -> Path:
    contract.validate_participant(participant)
    return Path(data_root) / "raw" / f"participant={participant}"


def session_json_path(data_root: Path | str, participant: str) -> Path:
    return participant_dir(data_root, participant) / "session.json"


def load_session(data_root: Path | str, participant: str) -> dict[str, Any] | None:
    p = session_json_path(data_root, participant)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_session(data_root: Path | str, participant: str,
                  session: dict[str, Any]) -> Path:
    pdir = participant_dir(data_root, participant)
    pdir.mkdir(parents=True, exist_ok=True)
    p = session_json_path(data_root, participant)
    p.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def init_session(data_root: Path | str, participant: str, *,
                 style: str | None = None,
                 styles: list[str] | None = None,
                 operator: str = "",
                 dry_run: bool = False) -> dict[str, Any]:
    """Create the participant scaffold + session.json.

    Idempotent: merges into an existing session.json.
    """
    contract.validate_participant(participant)
    plan_styles = styles or ([style] if style else list(contract.WALKING_STYLES))
    for s in plan_styles:
        contract.validate_style(s)

    existing = load_session(data_root, participant) or {}
    session: dict[str, Any] = {
        "schema_version": contract.METADATA_SCHEMA_VERSION,
        "participant": participant,
        "created_utc": existing.get("created_utc", _now_utc_iso()),
        "updated_utc": _now_utc_iso(),
        "styles": sorted(set(existing.get("styles", [])) | set(plan_styles)),
        "style_repeats": {s: contract.STYLE_REPEATS[s] for s in
                          sorted(set(existing.get("styles", [])) | set(plan_styles))},
        "operator": operator or existing.get("operator", ""),
    }

    if dry_run:
        return session

    write_session(data_root, participant, session)
    # Scaffold style sub-dirs so the layout is visible before any trial.
    for s in plan_styles:
        (participant_dir(data_root, participant) / f"style={s}").mkdir(
            parents=True, exist_ok=True)
    return session


