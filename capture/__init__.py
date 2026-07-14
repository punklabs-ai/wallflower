"""capture/ — thin, stdlib-only wrappers around external capture tools.

This package contains the *thin* glue that the per-node agents (nodes/) drive to
record raw CSI and BFI traces for the BFId reproduction lab (Todt, Morsbach,
Strufe, CCS '25). The heavy lifting lives in external tools:

  * CSI   -> PicoScenes (AX210 monitor capture, custom ``.raw`` records).
  * BFI   -> tcpdump (passive pcapng capture of IEEE 802.11 compressed
             beamforming report action/management frames).

Design rules (see project contract):
  * stdlib-only so it runs on bare perspective nodes (no numpy/torch here).
  * never crash when unprivileged: detect missing privilege/tools and print the
    exact command an operator should run, then degrade gracefully.
  * capture is always pinned to a known AP BSSID + channel; wildcard/
    promiscuous capture is refused.
  * when a real tool or privilege is missing, write no file and return the
    operator commands to run — never fabricate a capture artifact.

All file naming + directory layout is delegated to :mod:`wallflower.contract` via
:mod:`capture.naming` so it never drifts from the rest of the system.
"""

from __future__ import annotations

__all__ = ["naming", "csi_picoscenes", "bfi_pcap"]
