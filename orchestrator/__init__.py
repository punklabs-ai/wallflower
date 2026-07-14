"""wallflower orchestrator package.

Controller-side experiment session driver. Coordinates node-agents over SSH
(or locally in --profile pilot) to record synchronised CSI + BFI walking
traces, reproducing the BFId paper (Todt, Morsbach, Strufe, CCS '25).

"""
from __future__ import annotations

__all__ = ["cli", "ssh", "session", "clocks", "metadata"]
