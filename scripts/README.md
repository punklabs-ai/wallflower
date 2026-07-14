# scripts/ — wallflower operator scripts

Plain bash (`set -euo pipefail`) helpers an operator runs by hand when bringing
up or checking a node. They are **read-only or print-the-command** by design:
none of them change radio state, start daemons, or perform destructive ops, and
none *require* root (node1 sudo needs a password — privileged actions are
**printed** for the operator, never auto-run).

wallflower reproduces the BFId paper. Capture is pinned to a configured AP BSSID
and channel (see `configs/`); the recorder refuses wildcard capture.

Constants (feature dims, channels 37/85, roles) live in `wallflower/contract.py`; the
scripts mirror its values (`RADIO_ROLES`, `AP_CHANNELS`) rather than inventing new ones.

## check_radios.sh  (milestone-critical, works now, no root)
Inventories the Intel AX210 / Wi-Fi 6E radios a perspective node needs.

- Detects AX210s via `lspci` (matches `AX210` / `Wi-Fi 6E`).
- For each: PCI id, mapped netdev iface (via `/sys/class/net/*/device`, falls
  back to `iw dev`), phy, MAC, driver (`ethtool -i` or `/sys/.../uevent`),
  firmware, and 6 GHz / 160 MHz capability (`iw phy info`).
- **Confirms exactly two** radios; exits non-zero with guidance otherwise.
- Proposes the CSI vs BFI role split deterministically (lowest PCI slot → `csi`
  / AP-CSI ch37, next → `bfi` / AP-BFI ch85) — detected, not hardcoded.

```bash
scripts/check_radios.sh           # human table
scripts/check_radios.sh --json    # JSON blob (for the orchestrator)
```
Exit codes: `0` ok (2 radios) · `2` bad arg · `3` lspci missing · `4` wrong count.

## sync_clocks.sh
Inspects clock sync before a recording session — cross-node offset bounds the
CSI/BFI temporal alignment the parser can achieve.

- Detects a local daemon (chrony → ntpd → ptp4l) and shows offset
  (`chronyc tracking` / `ntpq -p` / `pmc`); prints install/setup guidance if none.
- Compares wall clocks across nodes (`--nodes` CSV, else concrete hosts from
  `configs/nodes.yaml`, skipping `*.lab.local` placeholders) by ssh-sampling
  `date +%s.%N`; reports max offset vs `--max-offset-ms` (default 5).
- Unreachable nodes are warned and skipped, never fatal.

```bash
scripts/sync_clocks.sh
scripts/sync_clocks.sh --nodes node1,persp2 --max-offset-ms 5
```
Exit codes: `0` within tolerance · `1` over tolerance · `2` bad arg.
**Note:** ssh-sampled wall clocks are *indicative*; true ms/sub-ms sync needs PTP.

## install_node.sh
Idempotent setup for a fresh perspective/recorder node (Ubuntu 26.04).

- `apt-get install` deps (`build-essential cmake dkms iw wireless-tools tcpdump
  tshark iperf3 chrony python3 python3-pip python3-venv git ethtool`), skipping
  already-present ones. If non-interactive sudo is unavailable it **prints** the
  exact `sudo apt-get …` command for the operator instead of hanging.
- Creates a venv and `pip install -e .[ml,capture]`.
- Prints NEXT STEPS that can't be auto-installed: **PicoScenes** (CSI), AX210
  **firmware** check, **monitor mode**, and **6 GHz regulatory domain** (`iw reg set`).

```bash
scripts/install_node.sh            # full
scripts/install_node.sh --no-apt   # venv + pip only (after operator runs apt)
```

## Typical order
```bash
scripts/install_node.sh      # one-time per node
scripts/check_radios.sh      # confirm 2x AX210 + roles
scripts/sync_clocks.sh       # confirm clocks before recording
```
