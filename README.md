# ParaFDEONet

Research code for **ParaFDEONet: A Unified Framework for Forward Prediction and Parameter Identification in Parameterized Functional Differential Equations**.

ParaFDEONet encodes history components and physical parameters in separate branches and uses a time trunk for forward prediction. The trained operator is frozen during parameter identification. This repository contains the five manuscript systems, controlled architecture comparisons, sensitivity studies, inverse solvers, and archived figure data.

[中文说明](README.zh-CN.md) · [Reproduction](docs/REPRODUCTION.md) · [Paper map](docs/PAPER_MAP.md) · [Data and weights](docs/ARTIFACTS.md) · [Validation](docs/VALIDATION.md)

## Start

Use Python 3.11 or 3.12. Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check
python run.py smoke competition
```

The smoke command generates a small synthetic dataset, trains small networks for a few steps, and evaluates them on CPU. It checks execution; it does not reproduce the paper's accuracy. Full experiments use the configurations and training seeds in `experiments/` and generally require a CUDA GPU.

```bash
python run.py smoke all
python run.py test all
python run.py train competition --device cuda:0 --cpus 16 --output-dir outputs/competition
```

Training and analysis outputs go to ignored `outputs/` directories. Weights and large training archives are not bundled; [artifact preparation](docs/ARTIFACTS.md) connects new formal training runs to the inverse and sensitivity benchmarks. Experiment tracking and notifications are disabled by default.

## Layout

```text
ParaFDE-ONet/
├── README.md
├── README.zh-CN.md
├── CITATION.cff
├── requirements.txt
├── requirements-optional.txt
├── run.py
├── experiments/
│   ├── competition/       # Two-state variable-delay competition
│   ├── sei/               # Three-state delayed epidemic system
│   ├── nicholson/         # Four-patch Nicholson system
│   ├── smart_grid/        # Eight-state delayed-grid screening
│   └── cstr/              # Delayed-cooling CSTR and operating-point search
├── benchmarks/
│   ├── inverse/           # Shared panels, DDE-LM, DDE-DE and frozen-operator LM
│   ├── sensitivity_jacobian/
│   └── feature_reuse/
├── scripts/               # Validation, artifact preparation and plotting
├── results/paper/         # Archived source data and manuscript figures
└── docs/                  # Protocols, provenance and validation
```

The original internal variant directories and model-format identifiers are retained for checkpoint compatibility. `run.py` selects the manuscript forward variants. `separate_parameter_shared` is the internal model key for ParaFDEONet. See the paper map for other historical method names.

## Inverse workflow

The current manuscript uses **ParaFDEONet-LM**. Its entry point is:

```bash
python run.py entry inverse-lm --help
```

The older `frozen` method in `entry inverse` uses Adam and remains available for the optimizer comparison. Use `--methods lm,de` there for the direct DDE baselines. The reproducibility guide specifies the shared-panel setup and history counts.

## Figures

The archived numerical sources can be inspected and plotted without training:

```bash
python run.py figures all
python scripts/refit_grid_gbt.py
```

Redrawn figures are written to `outputs/paper_figures/`. Plotting uses SciencePlots with `science`, `no-latex`, `grid`, and the **Source Han Serif** font. Install that font for matching typography.

## Citation

Author and manuscript information are recorded in `CITATION.cff`. A publication DOI has not been added because none was provided in the source materials. See [provenance](docs/PROVENANCE.md) for the source selection and release changes.

No license file was supplied with the selected source projects, so this package does not assign a new software license.
