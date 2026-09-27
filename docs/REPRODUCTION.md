# Reproduction

Run commands from the repository root. Use `python run.py list` to list entry points and `python run.py entry <name> --help` to inspect the original command-line options. Install the dependencies in `requirements.txt` first; a CUDA-enabled PyTorch installation is needed for formal GPU timing.

## Forward

```bash
python run.py train competition --device cuda:0 --cpus 16 --output-dir outputs/competition
python run.py train sei --device cuda:0 --cpus 16 --output-dir outputs/sei
python run.py train nicholson --device cuda:0 --cpus 16 --output-dir outputs/nicholson
python run.py train smart_grid --device cuda:0 --cpus 16 --output-dir outputs/smart_grid
python run.py train cstr --device cuda:0 --cpus 16 --output-dir outputs/cstr
```

The first three systems run 40,000, 60,000 and 60,000 configured training iterations respectively. Grid runs 80,000 and CSTR 40,000. The published full-size configurations are substantially larger than the smoke networks. Formal commands are provided for reproduction and were not run while assembling this source release.

To resume a forward run, pass the same `--output-dir` and `--resume`. Stage the three main-system artifacts using [the artifact guide](ARTIFACTS.md).

## Shared panels

This example reproduces the competition protocol. It generates 20 archives with eight histories each, then selects the first history for each inverse request. Direct solvers and the frozen LM must use the same archive and selected history.

```bash
python run.py entry inverse --system v2d --stage generate \
  --output-dir outputs/inverse_competition \
  --histories-per-panel 8 --histories-per-request 1 \
  --generation-processes 16 --cpus 16

python run.py entry inverse --system v2d --stage inverse \
  --methods lm,de --output-dir outputs/inverse_competition \
  --archive-dir outputs/inverse_competition/shared_panels \
  --histories-per-panel 8 --histories-per-request 1 \
  --lm-processes 16 --de-processes 16 --cpus 16 --resume

python run.py entry inverse-lm --system v2d \
  --archive-dir outputs/inverse_competition/shared_panels \
  --output-dir outputs/inverse_competition_lm \
  --histories-per-request 1 --gpus 0 --processes 1
```

| System | `entry inverse --system` | `entry inverse-lm --system` | Histories per request |
| --- | --- | --- | --- |
| Competition | `v2d` | `v2d` | 1 |
| SEI | `sei` | `d3d` | 8 |
| Nicholson | `n4d` | `n4d` | 1 |

Use the appropriate system identifier and separate output paths for SEI and Nicholson. Both methods use 50 cases per panel. Keep the archive at eight histories even when requesting one. The main text reports three of the four configured noise levels. The separate optimizer output contains `frozen_lm_per_case_results.csv`; the direct baseline output contains its own case and panel tables.

`entry inverse --methods frozen` invokes historical projected Adam. `entry inverse-compare` compares that optimizer with LM only when both runs use exactly matched archives and checkpoints.

## Sensitivity

Prepare both checkpoint variants and the test archive as described in `ARTIFACTS.md`, then run:

```bash
python run.py entry jacobian --device cuda:0 --reference-workers 16 \
  --network-batch-size 64 --output-dir outputs/jacobian
```

The script checks finite-difference and solver-step convergence, evaluates the paired Jacobians, and reports parameter-gradient directions. Its smoke option still requires trained checkpoints; the standalone forward smoke run does not supply paper-compatible models.

## Timing

```bash
python run.py entry feature-reuse --device cuda:0 --output outputs/feature_reuse

python run.py entry speed-competition \
  --checkpoint artifacts/competition/with_sensitivity/methods/separate_parameter_shared/seed_20261005/best_model.pt \
  --device cuda:0 --cpus 16 --output-dir outputs/speed_competition
```

Use `speed-sei` or `speed-nicholson` and the corresponding checkpoint for the other systems. Timing depends on hardware, thread counts and warm-up. The manuscript compares one RTX 4090 against AMD EPYC 7763 CPU solvers; a CPU smoke run does not reproduce those speed ratios.

## Grid

`run.py train smart_grid` trains all five configured seeds in one run. The fresh-history audit was originally written for two run directories. To use its interface without duplicating checkpoints, reproduce that split:

```bash
python run.py train smart_grid --seed 20261001 --device cuda:0 \
  --cpus 16 --output-dir outputs/grid_primary

python experiments/smart_grid/code/scripts/pipeline.py \
  --config experiments/smart_grid/configs/experiment.json \
  --seeds 20261002,20261003,20261004,20261005 \
  --devices 0 --processes 1 --cpus 16 \
  --wandb-mode disabled --no-ntfy --output-dir outputs/grid_additional

python run.py entry grid-fresh \
  --primary-run outputs/grid_primary --additional-run outputs/grid_additional \
  --output-dir outputs/grid_fresh --device cuda:0 --processes 16
```

The audit generates 512 new compatible histories, checks exact overlap against the original splits, solves the 64 parameter combinations, and evaluates all five checkpoints. Historical screening and multi-seed diagnostics are also available through `grid-screen` and `grid-multiseed`.

The manuscript GBT refit is available without neural-network weights:

```bash
python scripts/refit_grid_gbt.py
```

This fits the saved, training-only selected configuration to the archived training labels and checks its 64 predictions against the original results. The older `grid-gbt` entry uses an application-grid cross-validation protocol and is kept as an additional diagnostic.

## CSTR

After training CSTR, run the complete 20-event recalibration and operating-grid study:

```bash
python run.py entry cstr-online \
  --checkpoint outputs/cstr/forward/best_model.pt \
  --output-dir outputs/cstr_online --device cuda:0 --mode all
```

This uses the checked-in industrial configuration, including ParaFDEONet-LM, direct LM/DE, 10,201 operating points and direct-DDE verification. To inspect the archived result bundle:

```bash
python experiments/cstr/verify_results.py
python run.py figures cstr
```

## Figures

```bash
python run.py figures all
```

The plotting utilities read the archived source tables and write new files under `outputs/paper_figures/`. They do not replace the paper data or run training. Replotting source tables and retraining models are separate reproduction steps.
