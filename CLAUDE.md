# CLAUDE.md — wallflower

Guidance for Claude Code working in this repo — behavioural rules and
conventions. See `README.md` for the architecture and pipeline overview.

## What this is
A research reproduction of **BFId** (CCS '25 — WiFi identity inference from
Beamforming Feedback Information): capture the beamforming feedback WiFi devices
broadcast in the clear, parse it, and train an identity classifier.

## Non-negotiable rules
- **Real captured data only; never fabricate a result.** The parsers **RAISE**
  if a capture can't be decoded — there is no placeholder fallback — and capture
  writes no file when tcpdump/hardware is absent (it returns the operator
  commands instead). What's forbidden: letting mock output enter a dataset,
  capture log, or reported measurement **dressed up as real**. The live dashboard
  carries only real feeds — every router is an independent real capture (its
  own monitor interface + AP BSSID), and a router with no backing source shows
  as absent, never fabricated.
- **`wallflower/contract.py` is the single source of truth** (feature dims BFI=740 /
  CSI=212 +dt, naming, paths, `metadata.json`/`.npz` schema, styles, channels).
  Import from it; never redefine its constants.
- **Never store, echo, or hardcode the sudo password** anywhere (memory, files,
  scripts, command lines). Privileged access is scoped passwordless sudo; to add
  a binary, edit `/etc/sudoers.d/wallflower` (validate with `visudo -cf`), never
  use the password.
- **Surface privileged commands, don't force them.** Agents run privileged steps
  via `common.can_run_priv`/`run_priv` (scoped `sudo -n`) or return `ok=False`
  with a hint — they never block on `is_root()` and never auto-escalate.

## Conventions
- **Module seam:** stdlib-only for `orchestrator`/`nodes`/`capture` (they run on
  bare nodes; `pyyaml` ok on the controller). `numpy`/`torch`/`pandas` only in
  `parsers`/`models`. Use **`.venv/bin/python`** for anything importing them.
- Each agent action prints **exactly one** structured JSON object on stdout.
- Don't commit `data/**` or `.venv/` (gitignored).
- Verify changes by running the pipeline end-to-end on a real capture before
  claiming done. Tests run in script mode via `.venv/bin/python` (pytest isn't installed).

## Current reality (do not trust older prose that conflicts)
Code/config now encode the **operative** RF plan: one ASUS AP on **5 GHz ch36 /
80 MHz** (both roles), in `wallflower/contract.py` and `configs/ap_channels.yaml`. Host
`wlp1s0` does BFI + traffic + the RSSI dashboard; **CSI is captured bare-metal by
FeitCSI** on the second AX210. Everything runs on bare metal — no VMs. The paper's
**nominal** 6 GHz / ch37 + ch85 dual-AP plan is historical context only.
