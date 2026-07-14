"""BFI client node agent (radio B = legitimate beamformee / station).

In the BFId setup (Todt et al., CCS '25 sec. 5.1) the target's WiFi client is an
ordinary station ASSOCIATED to the AP. When the AP beamforms toward it, the
client returns Compressed Beamforming Reports (CBR); a separate passive recorder
(nodes.bfi_recorder_agent) observes those reports. This agent only makes radio B
behave as a normal client so the AP has someone to sound.

It associates one radio to one AP (AP-BFI). It does not deauth, does not spoof,
and refuses to join an unspecified/wildcard network.

  detect  -> identify the bfi-role radio + target AP-BFI network
  start   -> bring radio managed, (re)associate to AP-BFI, confirm association
  stop    -> disconnect (leave the network); clear pidfile
  status  -> report association state (SSID/BSSID/signal via `iw dev .. link`)
  health  -> driver/kernel/clock/tool summary for radio B

STDLIB-ONLY. Association needs root (wpa_supplicant / `iw connect`); when not
root the exact commands are printed and the agent exits non-fatally.
"""
from __future__ import annotations

import argparse
import os

from wallflower.contract import AP_BFI_CHANNEL, BAND_GHZ, WIDTH_MHZ

from . import common

AGENT = "bfi_client_agent"
ROLE = "bfi"


def _target_ap(args, bfi: dict) -> dict:
    """Resolve the AP-BFI network this client may join.

    Priority: --ssid/--bssid CLI > BFID_AP_BFI_SSID/BSSID env > lab.yaml
    'ap_bfi' block. A network must be specified; we never join a wildcard.
    """
    ssid = getattr(args, "ssid", None) or os.environ.get("BFID_AP_BFI_SSID")
    bssid = getattr(args, "bssid", None) or os.environ.get("BFID_AP_BFI_BSSID")
    if not ssid or not bssid:
        lab = common.load_yaml(common.LAB_YAML)
        ap = (lab.get("ap_bfi") or {}) if isinstance(lab.get("ap_bfi"), dict) else {}
        ssid = ssid or ap.get("ssid")
        bssid = bssid or ap.get("bssid")
    return {
        "ssid": ssid,
        "bssid": bssid,
        "channel": bfi.get("channel") or AP_BFI_CHANNEL,
        "iface": bfi.get("iface"),
    }


def _link_info(iface: str) -> dict:
    """Parse `iw dev <iface> link` to confirm association."""
    rc, out, _ = common.run(["iw", "dev", iface, "link"])
    info: dict = {"associated": False}
    if rc != 0:
        return info
    if out.strip().lower().startswith("not connected"):
        return info
    info["associated"] = "Connected to" in out
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Connected to"):
            info["bssid"] = s.split()[2]
        elif s.startswith("SSID:"):
            info["ssid"] = s.split(":", 1)[1].strip()
        elif s.startswith("signal:"):
            info["signal"] = s.split(":", 1)[1].strip()
        elif s.startswith("freq:"):
            info["freq"] = s.split(":", 1)[1].strip()
    return info


def detect(node: str | None) -> dict:
    bfi = common.resolve_role_iface(ROLE, node)
    radios = common.detect_radios()
    ok = bool(bfi["iface"])
    target = _target_ap(argparse.Namespace(ssid=None, bssid=None), bfi)
    return common.emit(
        AGENT, "detect", ok, node=bfi["node"], role=ROLE,
        bfi_radio=bfi, radios=radios,
        ap_channel=target["channel"], band_ghz=BAND_GHZ, width_mhz=WIDTH_MHZ,
    )


def start(args) -> dict:
    bfi = common.resolve_role_iface(ROLE, args.node)
    iface = bfi["iface"]
    if not iface:
        return common.emit(AGENT, "start", False, node=bfi["node"], role=ROLE,
                          error="no bfi-role AX210 radio resolved")
    target = _target_ap(args, bfi)

    # Refuse to join an unspecified network.
    if not target["ssid"] or not target["bssid"]:
        return common.emit(
            AGENT, "start", False, node=bfi["node"], iface=iface,
            error="refusing to associate without an explicit AP-BFI "
                  "target. Provide --ssid AND --bssid (or BFID_AP_BFI_SSID/"
                  "BFID_AP_BFI_BSSID, or an ap_bfi block in configs/lab.yaml).",
            target=target,
        )

    # Already associated to the right AP? Idempotent success.
    link = _link_info(iface)
    if link.get("associated") and link.get("bssid", "").lower() == target["bssid"].lower():
        common.write_pidfile(
            args_out_dir(args) or common.REPO_ROOT / "data", AGENT, os.getpid(),
            meta={"iface": iface, "associated": True, "bssid": target["bssid"],
                  "note": "pre-existing association",
                  "pre_existing_association": True},
        ) if args_out_dir(args) else None
        return common.emit(AGENT, "start", True, node=bfi["node"], iface=iface,
                          associated=True, target=target, link=link,
                          note="already associated")

    # Association is privileged. We use `iw connect` for an OPEN lab network and
    # surface a wpa_supplicant placeholder for a secured one.
    connect_cmds = [
        ["ip", "link", "set", iface, "down"],
        ["iw", "dev", iface, "set", "type", "managed"],
        ["ip", "link", "set", iface, "up"],
        # OPEN network association pinned to the lab BSSID + channel:
        ["iw", "dev", iface, "connect", target["ssid"], target["bssid"]],
    ]
    wpa_hint = (
        f"# secured lab AP: write a wpa_supplicant.conf with ssid=\"{target['ssid']}\" "
        f"bssid={target['bssid']}, then:\n"
        f"sudo wpa_supplicant -i {iface} -c /etc/wallflower/wpa_ap_bfi.conf -B"
    )

    # Association is privileged (iw/ip). Use root, or scoped passwordless sudo
    # (iw + ip are on the lab allow-list) for an OPEN lab AP. A secured AP needs
    # wpa_supplicant, which is NOT allow-listed -> surface the hint instead.
    if not all(common.can_run_priv(c) for c in connect_cmds):
        return common.emit(
            AGENT, "start", False, node=bfi["node"], iface=iface, role=ROLE,
            error="association requires root or scoped passwordless sudo for "
                  "iw/ip; run the printed commands then retry",
            target=target,
            privileged_commands=[common.privileged_hint(c) for c in connect_cmds],
            wpa_supplicant_hint=wpa_hint,
        )

    for cmd in connect_cmds:
        rc, _, err = common.run_priv(cmd, timeout=15)
        if rc != 0:
            return common.emit(AGENT, "start", False, node=bfi["node"], iface=iface,
                              error=f"{' '.join(common.priv_cmd(cmd))} failed: {err.strip()}",
                              target=target)
    link = _link_info(iface)
    ok = bool(link.get("associated"))
    out_dir = args_out_dir(args)
    if ok and out_dir:
        common.write_pidfile(out_dir, AGENT, os.getpid(),
                            meta={"iface": iface, "associated": True,
                                  "bssid": target["bssid"]})
    return common.emit(AGENT, "start", ok, node=bfi["node"], iface=iface,
                      associated=ok, target=target, link=link,
                      error=None if ok else "association did not complete")


def stop(args) -> dict:
    bfi = common.resolve_role_iface(ROLE, args.node)
    iface = bfi["iface"]
    out_dir = args_out_dir(args)
    pidinfo = common.read_pidfile(out_dir, AGENT) if out_dir else None

    # Idempotent no-op: no pidfile, or a pre-existing association we did not
    # create -> nothing to disconnect, that's a clean stop.
    if not pidinfo or pidinfo.get("pre_existing_association"):
        if out_dir:
            common.pidfile_path(out_dir, AGENT).unlink(missing_ok=True)
        mode = "pre_existing_association" if pidinfo and pidinfo.get("pre_existing_association") else "no_pidfile"
        return common.emit(AGENT, "stop", True, node=bfi["node"], iface=iface,
                          mode=mode, note="no association to tear down")

    disconnect = ["iw", "dev", iface, "disconnect"] if iface else []
    if not iface:
        return common.emit(AGENT, "stop", False, node=bfi["node"],
                          error="no bfi-role radio resolved")
    if not common.can_run_priv(disconnect):
        # Remove pidfile bookkeeping but surface the privileged disconnect.
        if out_dir:
            common.pidfile_path(out_dir, AGENT).unlink(missing_ok=True)
        return common.emit(AGENT, "stop", False, node=bfi["node"], iface=iface,
                          error="disconnect requires root or scoped sudo for iw",
                          privileged_commands=[common.privileged_hint(disconnect)])
    rc, _, err = common.run_priv(disconnect)
    if out_dir:
        common.pidfile_path(out_dir, AGENT).unlink(missing_ok=True)
    ok = rc == 0
    return common.emit(AGENT, "stop", ok, node=bfi["node"], iface=iface,
                      error=None if ok else err.strip())


def status(args) -> dict:
    bfi = common.resolve_role_iface(ROLE, args.node)
    iface = bfi["iface"]
    link = _link_info(iface) if iface else {"associated": False}
    return common.emit(AGENT, "status", True, node=bfi["node"], iface=iface,
                      associated=link.get("associated", False), link=link,
                      iface_type=common._iw_dev_types().get(iface, "unknown"))


def health(args) -> dict:
    bfi = common.resolve_role_iface(ROLE, args.node)
    return common.emit(AGENT, "health", True, node=bfi["node"], role=ROLE,
                      bfi_radio=bfi, kernel=common.kernel_info(),
                      clock=common.clock_sources(), tools=common.tool_availability(),
                      radios=common.detect_radios())


def args_out_dir(args):
    try:
        return common.default_out_dir(args)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m nodes.bfi_client_agent")
    common.add_common_args(parser)
    parser.add_argument("--ssid", help="AP-BFI SSID to join")
    parser.add_argument("--bssid", help="AP-BFI BSSID to join")
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
