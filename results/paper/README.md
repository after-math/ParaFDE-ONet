# Paper results

These are archived outputs from the supplied projects. Their source identities are recorded in the [source manifest](../../docs/source_manifest.json).

| Directory | Contents |
| --- | --- |
| `source_data/` | Forward and inverse source tables, timing, training curves, grid predictions and supplementary PINN-DDE spreadsheet |
| `figures/` | Figures referenced by the supplied main text and supplementary information |
| `cstr/` | Forward summaries, 252 inversion records, 20 event records and the representative operating map |
| `sensitivity/` | Per-case Jacobians, convergence checks and gradient-direction summaries |
| `smart_grid/` | Training-only aggregate labels and the GBT tuning-result archive |

Use `python run.py figures all` from the repository root to redraw the supported figures. The newly generated files are written to `outputs/paper_figures/`.

The main-text inverse source is `source_data/frozen_lm_inverse_summary.csv`, with per-case and per-panel files containing `frozen_lm` in their names. Other inverse files are retained for supplementary and historical comparisons. The competition forward illustration was recovered from a saved SVG, as stated in its filename and the paper map.
