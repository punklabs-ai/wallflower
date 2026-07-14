"""Build, write and validate the canonical per-trial metadata.json.

Uses contract.TrialMetadata + contract.RadioRecord so the schema never drifts.
The orchestrator writes ONE metadata.json per trial into
contract.raw_trial_dir(...). validate-session reads it back and checks it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wallflower import contract

from .session import LabConfig, NodeInfo


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _radio_record_from_cfg(node: NodeInfo, radio: dict[str, Any],
                           override_role: str | None = None) -> contract.RadioRecord:
    return contract.RadioRecord(
        role=override_role or radio.get("role", ""),
        node=node.node,
        iface=radio.get("iface", ""),
        phy=radio.get("phy", ""),
        mac=radio.get("mac", ""),
        pci=radio.get("pci", ""),
        perspective=radio.get("perspective"),
        ap=radio.get("ap", ""),
        channel=radio.get("channel"),
    )


def collect_radios(cfg: LabConfig, perspectives: list[int],
                   health: dict[str, dict[str, Any]] | None = None
                   ) -> list[contract.RadioRecord]:
    """Build the radios[] list for metadata from the resolved config.

    Includes the per-perspective CSI radio, the per-perspective BFI client
    radio, and the central passive BFI recorder radio. `health` (node->agent
    health JSON) may refine iface/mac if a node reports different facts at
    record time.
    """
    radios: list[contract.RadioRecord] = []
    seen: set[tuple[str, str, int | None]] = set()

    for p in perspectives:
        node = cfg.csi_node_for(p)
        if not node:
            continue
        for role in ("csi", "bfi"):
            radio = cfg.radio_for_role(node, role, perspective=p)
            if not radio:
                # In pilot the single radio set may not be tagged per-perspective.
                radio = cfg.radio_for_role(node, role)
            if not radio:
                continue
            rec = _radio_record_from_cfg(node, radio, override_role=role)
            rec.perspective = p
            key = (rec.role, rec.mac, rec.perspective)
            if key in seen:
                continue
            seen.add(key)
            radios.append(rec)

    rec_node = cfg.recorder_node()
    if rec_node:
        rradio = (cfg.radio_for_role(rec_node, "bfi_recorder")
                  or cfg.radio_for_role(rec_node, "bfi"))
        if rradio:
            rec = _radio_record_from_cfg(rec_node, rradio, override_role="bfi_recorder")
            rec.perspective = None
            rec.ap = "AP-BFI"
            rec.channel = rradio.get("channel") or contract.AP_BFI_CHANNEL
            radios.append(rec)

    # Optionally fold in health-reported corrections (best-effort).
    if health:
        for rec in radios:
            h = health.get(rec.node)
            if not h:
                continue
            # node-agent health may report observed radios under 'radios'
            for hr in (h.get("radios") or []):
                if hr.get("role") == rec.role and hr.get("iface"):
                    rec.iface = hr.get("iface", rec.iface)
                    rec.mac = hr.get("mac", rec.mac)
                    rec.phy = hr.get("phy", rec.phy)
    return radios


def build_capture_plan(cfg: LabConfig, perspectives: list[int],
                       radios: list[contract.RadioRecord]) -> dict[str, Any]:
    """Per-modality capture summary + honest node radio-availability limits.

    BFId records CSI and BFI SIMULTANEOUSLY in one session; this block makes the
    CSI vs BFI radio/role/channel assignment explicit for time alignment, and
    documents what THIS deployment's radios can/can't do concurrently. On the
    one-node pilot (node1: only 2x AX210, single on-air AP LAB_AP on ch36) the
    same physical radios are reused across the CSI / BFI / recorder roles, so a
    REAL simultaneous CSI+BFI capture is radio-constrained — see `limits`.
    """
    def _radio_dict(rec: contract.RadioRecord) -> dict[str, Any]:
        return {"role": rec.role, "node": rec.node, "iface": rec.iface,
                "phy": rec.phy, "mac": rec.mac, "ap": rec.ap,
                "channel": rec.channel, "perspective": rec.perspective}

    by_role: dict[str, list[dict[str, Any]]] = {}
    for rec in radios:
        by_role.setdefault(rec.role, []).append(_radio_dict(rec))

    # Distinct physical radios (by mac) actually engaged this trial.
    macs: dict[str, set[str]] = {}
    for rec in radios:
        macs.setdefault(rec.mac, set()).add(rec.role)
    n_radios = len([m for m in macs if m])
    contended = {m: sorted(r) for m, r in macs.items() if m and len(r) > 1}

    plan: dict[str, Any] = {
        "concurrent": True,   # CSI + BFI launched together (one trial / session)
        "modalities": {
            "csi": {
                "feature_dim": contract.CSI_FEATURES,
                "nominal_rate_hz": contract.NOMINAL_RATE_HZ["csi"],
                "ap": "AP-CSI",
                "channel": contract.AP_CSI_CHANNEL,
                "radios": by_role.get("csi", []),
            },
            "bfi": {
                "feature_dim": contract.BFI_FEATURES,
                "nominal_rate_hz": contract.NOMINAL_RATE_HZ["bfi"],
                "ap": "AP-BFI",
                "channel": contract.AP_BFI_CHANNEL,
                "client_radios": by_role.get("bfi", []),
                "recorder_radios": by_role.get("bfi_recorder", []),
            },
        },
        "distinct_physical_radios": n_radios,
    }

    # Honest, machine-readable limits for this deployment (Research B).
    limits: list[str] = []
    if contended:
        for mac, roles in contended.items():
            limits.append(
                f"radio {mac} is shared across roles {roles}: a REAL capture "
                "cannot drive these roles simultaneously on one radio "
                "(monitor vs managed are mutually exclusive).")
    if n_radios <= 2:
        limits.append(
            "node has <=2 AX210 radios: simultaneous REAL CSI (monitor) + BFI "
            "client + passive BFI recorder needs 3 independent radios; with 2 "
            "radios + 1 on-air AP (LAB_AP ch36) real-both is NOT achievable. "
            "Real CSI is additionally driver-gated on kernel 7.0 (FeitCSI/"
            "PicoScenes have no kernel-7.0 build), so CSI cannot be captured on "
            "this host; BFI may run real. The orchestration here still fans both "
            "out CONCURRENTLY "
            "and records this limit; it never swaps/unloads the live iwlwifi "
            "driver or touches the dashboard monitor.")
    plan["limits"] = limits
    if contended:
        plan["radio_contention"] = contended
    return plan


def build_trial_metadata(cfg: LabConfig, *, participant: str, style: str,
                         trial: str, direction: str, perspectives: list[int],
                         clock_sync: dict[str, Any],
                         health: dict[str, dict[str, Any]] | None = None,
                         notes: str = "",
                         capture_start_epoch: float | None = None
                         ) -> contract.TrialMetadata:
    contract.validate_participant(participant)
    contract.validate_style(style)
    contract.validate_trial(trial)
    for p in perspectives:
        contract.validate_perspective(p)
    if direction not in contract.DIRECTIONS:
        raise ValueError(f"direction {direction!r} not in {contract.DIRECTIONS}")

    radios = collect_radios(cfg, perspectives, health=health)
    cap_iso = (datetime.fromtimestamp(capture_start_epoch, tz=timezone.utc).isoformat()
               if capture_start_epoch is not None else "")
    return contract.TrialMetadata(
        schema_version=contract.METADATA_SCHEMA_VERSION,
        participant=participant,
        trial=trial,
        style=style,
        direction=direction,
        perspectives=list(perspectives),
        timestamp_utc=_now_utc_iso(),
        radios=radios,
        clock_sync=clock_sync,
        notes=notes,
        capture_start_epoch=capture_start_epoch,
        capture_start_utc=cap_iso,
        capture_plan=build_capture_plan(cfg, perspectives, radios),
    )


def write_trial_metadata(data_root: Path | str, meta: contract.TrialMetadata) -> Path:
    trial_dir = contract.raw_trial_dir(data_root, meta.participant, meta.style, meta.trial)
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "logs").mkdir(exist_ok=True)
    path = trial_dir / contract.metadata_name()
    path.write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=False) + "\n",
                    encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Validation (validate-session)
# --------------------------------------------------------------------------- #
_REQUIRED_META_KEYS = {
    "schema_version", "participant", "trial", "style", "direction",
    "perspectives", "timestamp_utc", "band_ghz", "width_mhz",
    "ap_channels", "radios", "clock_sync",
}


def validate_metadata_dict(meta: dict[str, Any]) -> list[str]:
    """Return a list of schema problems (empty == valid)."""
    errors: list[str] = []
    missing = _REQUIRED_META_KEYS - set(meta)
    if missing:
        errors.append(f"metadata missing keys: {sorted(missing)}")
    sv = meta.get("schema_version")
    if sv != contract.METADATA_SCHEMA_VERSION:
        errors.append(
            f"schema_version {sv!r} != expected {contract.METADATA_SCHEMA_VERSION!r}")
    try:
        if meta.get("participant"):
            contract.validate_participant(meta["participant"])
        if meta.get("style"):
            contract.validate_style(meta["style"])
        if meta.get("trial"):
            contract.validate_trial(meta["trial"])
    except ValueError as exc:
        errors.append(str(exc))
    for p in meta.get("perspectives", []) or []:
        if p not in contract.PERSPECTIVES:
            errors.append(f"perspective {p} not in {contract.PERSPECTIVES}")
    if meta.get("band_ghz") not in (None, contract.BAND_GHZ):
        errors.append(f"band_ghz {meta.get('band_ghz')} != {contract.BAND_GHZ}")
    if meta.get("width_mhz") not in (None, contract.WIDTH_MHZ):
        errors.append(f"width_mhz {meta.get('width_mhz')} != {contract.WIDTH_MHZ}")
    return errors
