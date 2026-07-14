# wallflower

A WiFi-sensing lab that reproduces the BFId paper — identity inference from the
beamforming feedback 802.11 devices broadcast in the clear:

> Todt, Morsbach, Strufe. *"BFId: Identity Inference Attacks Utilizing
> Beamforming Feedback Information."* CCS '25.

wallflower

1. records synchronised **CSI + BFI** traces from a 4-perspective setup,
2. segments and parses them into ML-ready variable-length time series, and
3. trains/evaluates a baseline **LSTM identity classifier**.

The headline paper result it targets is **BFI identity accuracy 99.5% ± 0.38**
from normal walking; the next milestone is reproducing it on our own captures and
showing it generalises beyond one room.

---

## What is BFId?

802.11ac/ax/be beamforming requires a receiver to periodically send the
transmitter **Beamforming Feedback Information (BFI)** — compressed
(quantised-angle) channel-state matrices — *in the clear*, even on encrypted
networks. BFId shows that a passive observer who records this BFI while a person
walks can infer **who** the person is, because gait perturbs the channel in a
person-specific way. wallflower reproduces this by capturing BFI alongside
ground-truth **CSI** (channel state information) from multiple perspectives and
training an identity classifier.

Feature dimensions (from the paper, defined once in `wallflower/contract.py`):

| Modality | Features | Composition | Nominal rate |
|----------|----------|-------------|--------------|
| BFI | 740 (+1 dt) | 10 quantised angles × 74 channels | ~10 Hz |
| CSI | 212 (+1 dt) | (phase + magnitude) × 53 subcarriers × 2 antennas | ~285 Hz |

The `+1` is an appended time-delta column the models consume; parsers store it
separately as `dt` in each `.npz`.

---

## Physical topology

> **Operative setup (current pilot).** A single lab-owned **ASUS RT-AXE7800** AP
> on **5 GHz / channel 36 / 80 MHz** (SSID `LAB_AP`, BSSID `AA:BB:CC:DD:EE:F4`)
> serves both the CSI and BFI roles — this is what `configs/ap_channels.yaml` and
> `wallflower/contract.py` actually encode. On node1 the host `wlp1s0` AX210 does **BFI
> capture + traffic + the live RSSI dashboard**; **CSI is captured bare-metal by
> FeitCSI** on the second AX210. Everything runs on bare metal — no VMs. Real BFI
> is captured and decoded. The 6 GHz / dual-AP plan described below is the paper's
> **nominal** full-deployment target, not the current rig.

Paper-nominal full 4-perspective deployment (logical inventory in
`configs/nodes.yaml` under `topology.full`):

- **2× access points**, both 6 GHz / 160 MHz:
  - **AP-CSI** on **channel 37** — carries the CSI-driving traffic.
  - **AP-BFI** on **channel 85** — its beamforming sounding is what the passive
    recorder collects.
- **4 perspective nodes**, each with **2× Intel AX210/AX1675 2×2** radios:
  - **radio A = CSI** capture (monitor mode; FeitCSI).
  - **radio B = BFI** client (associates to AP-BFI to elicit sounding).
- **1 CSI-traffic node** — generates `iperf3` traffic (200 Mb/s TCP drives BFI
  sounding; 30 Kb/s UDP keeps CSI flowing).
- **1 passive BFI recorder** — a single monitor-mode capture (`tcpdump`) of all
  BFI sounding into one `bfi_recorder.pcapng`.
- **1 controller** — orchestrates sessions over SSH and (optionally) trains.

```
              AP-CSI (ch37)            AP-BFI (ch85)        6 GHz / 160 MHz
                  |                         |
   csi-traffic ---+                         +--- bfi-recorder (passive, 1 pcapng)
                  |                         |
        +---------+----------+   +----------+----------+   ... x4 perspectives
        | perspective node N |   | radioA=CSI monitor  |
        | 2x AX210           |   | radioB=BFI client   |
        +--------------------+   +---------------------+
                         \             /
                          \           /
                        controller (SSH orchestrator + trainer)
```

### Node-agent invocation contract

The controller talks to each node over SSH (or locally in pilot) via:

```
python3 -m nodes.<agent> <action> --participant P001 --style normal --trial 001 \
    [--perspective N] [--out-dir DIR]
# actions: detect | start | stop | status | health
```

Every agent prints **one structured JSON object** to stdout
(`{agent, action, ok, node, ts_utc, ...}`). `start` writes a pidfile so `stop`
can terminate the capture. Output filenames come from `wallflower.contract`
(`csi_raw_name(p)`, `bfi_recorder_name()`).

### Canonical raw layout (per trial)

```
data/raw/participant=P001/style=normal/trial=001/
    metadata.json
    csi_p1.raw  csi_p2.raw  csi_p3.raw  csi_p4.raw
    bfi_recorder.pcapng
    logs/
```

---

## Repository layout

```
wallflower/
├── wallflower/            # shared CONTRACT (constants, paths, dataclasses) — import this
│   └── contract.py  #   single source of truth: feature dims, channels, layout
├── orchestrator/    # controller CLI (`wallflower`) — session/trial orchestration over SSH
├── nodes/           # per-node agents (csi, bfi client, bfi recorder, traffic)
├── capture/         # low-level capture helpers (tcpdump/iw wrappers, stdlib-only)
├── parsers/         # CSI .raw + BFI .pcapng -> variable-length .npz / parquet
├── models/          # baseline LSTM identity classifier + training/eval
├── configs/         # lab.yaml, nodes.yaml, ap_channels.yaml
├── scripts/         # operator convenience scripts
├── experiments/     # experiment configs / run outputs
└── data/            # raw / processed / models / reports  (git-ignored; .gitkeep)
```

Config files (kept consistent with `wallflower/contract.py`):

- `configs/lab.yaml` — data root, 80/20 split, perspectives, styles + repeats,
  band/width/channels, clock-sync tolerance, sample-rate targets, traffic.
- `configs/nodes.yaml` — full 4-perspective inventory **and** the `pilot`
  profile mapping every role onto node1.
- `configs/ap_channels.yaml` — operative RF plan: single ASUS AP, 5 GHz / 80 MHz,
  both roles on ch36 (SSID `LAB_AP`, BSSID `AA:BB:CC:DD:EE:F4`), with the
  paper-nominal 6 GHz / dual-channel plan noted as historical context.

---

## Install

```bash
# Core (orchestrator + nodes + capture stay stdlib-first; only pyyaml needed):
pip install -e .

# ML stack (parsers + models): numpy/torch/pandas/pyarrow/scikit-learn/optuna
pip install -e ".[ml]"

# BFI pcapng parsing helper (scapy):
pip install -e ".[capture]"

# Dev tooling (ruff + pytest):
pip install -e ".[dev]"
```

Requires Python ≥ 3.12.

### node1 hardware facts (the pilot machine)

- Ubuntu 26.04 LTS, kernel 7.0.0, Python 3.14; `iw` 6.17, `tcpdump`, `ssh` present.
- **2× Intel AX210/AX1675 2×2** radios (iwlwifi loaded, firmware present):
  - PCI `01:00.0` → `wlp1s0`, `phy1`, MAC `AA:BB:CC:DD:EE:02` → **BFI client / traffic** role.
  - PCI `02:00.0` → `wlp2s0`, `phy2`, MAC `AA:BB:CC:DD:EE:03` → **BFI recorder** role.
- Wired control plane: `eno1`.
- **Assumed MISSING** (degrade gracefully / document install): `tshark`,
  `iperf3`, `chrony`/`ntp`, `ptp4l`, PicoScenes, `git`, `gcc`/`make`/`cmake`/`dkms`.

### sudo / privilege

`sudo` on node1 **requires a password** — non-interactive root is not available.
Anything needing root (monitor mode, channel set via `iw`, package install, raw
socket capture) is **printed for the operator to run**, prefixed clearly, e.g.:

```
[OPERATOR-RUN] sudo iw dev wlp1s0 set type monitor
[OPERATOR-RUN] sudo iw dev wlp1s0 set channel 36 80MHz   # 5 GHz
```

Read-only inspection (`lspci`, `iw dev`, `ip link`, reading `/sys`) works
without root.

---

## Quickstart — ONE-NODE PILOT

The pilot collapses every role onto node1 (see `configs/nodes.yaml` profile
`pilot`). It validates the end-to-end pipeline on a single machine before scaling
to 4 perspectives.

```bash
# 0. Inspect radios without root (sanity check the pilot mapping):
python3 -m nodes.csi_agent detect --perspective 1

# 1. Create a session (writes session config under data/):
wallflower init-session --participant P001 --profile pilot

# 2. Start a trial (clock-sync check, spawns node agents, writes metadata.json):
wallflower start-trial --participant P001 --style normal --trial 001

# ...participant walks back and forth...

# 3. Stop the trial (terminates captures via pidfiles, finalises metadata):
wallflower stop-trial --participant P001 --style normal --trial 001

# 4. Validate everything that was recorded for the session:
wallflower validate-session --participant P001
```

Any privileged step that cannot run will print an `[OPERATOR-RUN]` command for
you to execute manually, then continue without crashing.

---

## The `wallflower` CLI

Installed as the `wallflower` console script (`orchestrator.cli:main`):

| Command | Purpose |
|---------|---------|
| `init-session` | Create/validate a session and select a profile (`pilot` or `full`). |
| `start-trial` | Run the clock-sync gate, spawn node agents (CSI, BFI client, BFI recorder, traffic), write the trial `metadata.json`. |
| `stop-trial` | Stop captures via their pidfiles and finalise `metadata.json`. |
| `validate-session` | Check raw layout, file presence, sample-rate / clock-sync tolerances (per `configs/lab.yaml`) and report problems. |

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
