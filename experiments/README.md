# wallflower — Reproduction Experiment Configs

Reproduction of BFId (Todt, Morsbach, Strufe,
*"BFId: Identity Inference Attacks Utilizing Beamforming Feedback Information"*,
CCS '25). The empty-room control means an *unoccupied* room.

These YAML files are declarative experiment definitions consumed by
`models/train.py` and `models/evaluate.py`. They never redefine constants from
`wallflower/contract.py`; feature dimensions, label vocabularies, walking styles,
perspectives and paths all come from the contract via the models stack.

## Shared schema

All six configs use the same top-level keys so the runners can treat them
uniformly:

| Key            | Meaning                                                                              |
|----------------|--------------------------------------------------------------------------------------|
| `name`         | Run identifier; artefacts land under `data/models/<name>/`.                          |
| `description`  | Human-readable summary.                                                              |
| `modality`     | `bfi` or `csi` (sets `input_dim` via `contract.expected_feature_dim(..., dt=True)`). |
| `data`         | Sample filters: `participants` (`null`=all), `styles`/`perspectives`, plus transfer extras `train_styles`/`test_styles` where relevant. Map onto `models.dataset.load_index(...)`. |
| `split`        | `train_frac` (0.8 paper baseline), `seed`, `repeats:5` (mean +- std over seeded splits). `test_frac = 1 - train_frac` for `stratified_split`. |
| `model`        | `hidden`, `lstm_layers`, `fc_hidden`, `dropout` -> `models.lstm_classifier.BFIdLSTM`.|
| `optim`        | `lr`, `weight_decay`, `batch_size`, `epochs` (Adam, per paper sec. 5.2).             |
| `standardise`  | `true` -> fit `models.dataset.Standardiser` on the **train split only**.             |
| `metric`       | `top1` (identity accuracy) or `top2` (empty-room control).                           |
| `ablation`     | Present only for sweeps (`sample_rate_ablation`, `perspective_ablation`): the swept parameter and its grid. |
| `paper_reference` / `expected_result` | Provenance + the number/shape we expect to reproduce.         |

> **Schema note for integrators.** An earlier draft of
> `bfi_identity_baseline.yaml` used a different layout (`filter`, `split.test_frac`,
> a `train:` block, `data_root`, `optuna`). All six configs here are normalised to
> the schema above per the module brief. `models/train.py` (not yet written) is the
> single place that translates this schema into dataset/model/optimiser calls:
> it should read `split.train_frac` and pass `test_frac = 1 - train_frac` to
> `models.dataset.stratified_split`, loop `split.repeats` over derived seeds, and map
> `optim.*`/`model.*` straight through. `data_root` is taken from `configs/lab.yaml`
> or a CLI flag, not from these experiment files.

## Config -> paper figure / hypothesis map

| Config                          | Modality | Regime                                   | Paper ref            | Expected result                         |
|---------------------------------|----------|------------------------------------------|----------------------|-----------------------------------------|
| `bfi_identity_baseline.yaml`    | bfi      | normal walking, all 4 perspectives pooled| Headline (sec. 5.3)  | top-1 ~**99.5% +- 0.38**                |
| `csi_identity_baseline.yaml`    | csi      | normal walking, all perspectives pooled  | CSI vs BFI (sec. 5.3)| top-1 ~**82.4%**                        |
| `sample_rate_ablation.yaml`     | bfi      | decimate packets/sample over a factor grid | **Fig. 3**         | BFI robust; mild degradation at low rate|
| `walking_style_transfer.yaml`   | bfi      | train normal, test backpack/crate/fast/turnstile (no retrain) | **Fig. 4** | accuracy drops, reported per style |
| `perspective_ablation.yaml`     | bfi      | matched diagonal + cross-perspective 4x4 matrix | **Fig. 5/6**  | matched high, cross-perspective degrades|
| `empty_room_control.yaml`       | bfi      | train normal, test empty-room, metric top-2 | **H7** (sec. 5.3) | ~**2.34% top-2** (near chance)          |

## Sweep / transfer details

- **`sample_rate_ablation`** — `ablation.subsample_factors: [1,2,3,4,6,8,10]`.
  Factor `f` keeps every `f`-th timestep (and re-accumulates `dt`), lowering the
  effective rate off `contract.NOMINAL_RATE_HZ["bfi"] = 10 Hz`. One train+eval per
  factor; report accuracy vs factor (paper Fig. 3).
- **`walking_style_transfer`** — `data.train_styles=[normal]`,
  `data.test_styles=[backpack,crate,fast,turnstile]`. Fit model + Standardiser on
  normal only; reuse the fixed participant label vocabulary to score each held-out
  style with `metrics.per_group_accuracy` keyed on style.
- **`perspective_ablation`** — `ablation.perspectives=[1,2,3,4]`,
  `cross_perspective: true` builds the full 4x4 train/test matrix (matched diagonal
  = Fig. 5, off-diagonal mismatch = Fig. 6). Set `false` for the matched-only
  diagonal.
- **`empty_room_control`** — needs dedicated **empty-room captures** (same rig, no
  participant in the path), tagged either as `style == "empty_room"` (preferred) or
  via a reserved `empty_participant_tag` (e.g. `P000`). Scored with `metric: top2`.

## Primary success criteria (from the project brief)

These are the acceptance gates the experiment suite exists to demonstrate:

1. **Pilot (1 perspective)** — the rig + parser + trainer run end-to-end on a
   single perspective. Filter any config to one perspective (e.g.
   `data.perspectives: [1]`) to exercise the pilot path.
2. **Full (4 perspectives)** — the same pipeline scales to all four perspectives;
   `bfi_identity_baseline` pools `perspectives: [1,2,3,4]`.
3. **BFI parser produces valid variable-length samples** — every consumed `.npz`
   has `x` of shape `[T, contract.BFI_FEATURES]` (740) with a matching per-step
   `dt`; the dataset appends `dt` to give `input_dim = 741`.
4. **Training on BFI-only** — the headline (`bfi_identity_baseline`) and all
   sweeps default to `modality: bfi`; `csi_identity_baseline` is the contrast.
5. **Evaluation by participant / style / perspective** — produced via
   `models.metrics.per_group_accuracy`; the transfer and perspective configs make
   the style/perspective breakdowns first-class, and `split.repeats: 5` yields the
   paper-style mean +- std reporting.

## Running

```bash
# Train (writes model + standardiser + label vocab + run report under data/models/<name>/)
python3 -m models.train    --config experiments/bfi_identity_baseline.yaml

# Evaluate with paper-style breakdowns (top-1, per-participant/style/perspective)
python3 -m models.evaluate --config experiments/bfi_identity_baseline.yaml
```

The `[ml]` optional dependency group (numpy, torch, pandas, pyarrow,
scikit-learn) must be installed for the models stack; the capture / orchestrator
/ node agents stay stdlib-only and do not import these configs.
