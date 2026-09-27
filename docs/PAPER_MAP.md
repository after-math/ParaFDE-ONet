# Paper map

The scope follows the shortened `FDE_parameter.tex` and `Supplementary_Information.tex` in the source article workspace. Older exploratory systems are excluded.

| Study | Manuscript implementation | Main configuration |
| --- | --- | --- |
| Competition forward | `experiments/competition/four_methods_sensitivity_5seeds/` | `configs/experiment.json` within that directory |
| SEI forward | `experiments/sei/stratified_history_balanced_sensitivity/` | `configs/experiment.json` |
| Nicholson forward | `experiments/nicholson/four_methods_normalized_sensitivity_5seeds/` | `configs/experiment.json` |
| Grid forward and screening | `experiments/smart_grid/` | `configs/experiment.json`, `configs/survivability_experiment.json` |
| CSTR recalibration and search | `experiments/cstr/` | `configs/experiment.json`, `configs/industrial.json` |
| ParaFDEONet-LM | `benchmarks/inverse/frozen_lm_ablation/` | `config.json` |
| DDE-LM and DDE-DE | `benchmarks/inverse/code/` | `benchmarks/inverse/configs/` |
| Sensitivity Jacobian | `benchmarks/sensitivity_jacobian/` | `configs/benchmark.json` |
| Feature reuse | `benchmarks/feature_reuse/` | `configs/benchmark.json` |
| Forward timing | `experiments/{competition,sei,nicholson}/separate_parameter_speed_benchmark/` | `configs/benchmark.json` |

## Names

| Manuscript name | Historical code key |
| --- | --- |
| ParaFDEONet | `separate_parameter_shared`; some output labels use `PFDEONet` or `Separate-Parameter MIONet` |
| JI-DeepONet | `single_branch_deeponet` |
| RP-MIONet | `two_branch_mionet`, `three_branch_mionet`, or `four_branch_mionet` |
| State-specific parameter branch | `separate_parameter_state_specific` |
| Grid ParaFDEONet | `parafdeonet` |
| ParaFDEONet-LM | `frozen_lm` |
| Historical frozen Adam | `frozen` |

## Protocol

Forward training uses five seeds for competition, SEI, Nicholson and grid, and one CSTR seed. The competition branch takes physical parameter values; the SEI and Nicholson branches normalize their parameter inputs. The corresponding source variants are retained exactly as separate model formats.

The shared-panel inverse study uses 20 panels and 50 parameter cases per panel. Competition and Nicholson use one selected history per request; SEI uses eight histories. The configuration also retains the 0.010 noise level, while the current main text reports 0.005, 0.020 and 0.050. Aggregation is over panel means, with sample standard deviations across the 20 panels.

Sensitivity-Jacobian comparisons condition on a fixed checkpoint pair per system: competition seed 20261005, SEI seed 20260903 and Nicholson seed 20260901. They are not five-training-seed confidence intervals. Feature-reuse timing instead uses Nicholson seed 20260905 for all three architectures.

The `three_method_inverse_comparison/` directories are historical supplementary pipelines, including PINN-DDE feasibility and early inverse variants. Their method defaults should not replace the current shared-panel LM protocol.

## Results

`results/paper/source_data/frozen_lm_inverse_summary.csv` and the three `*_frozen_lm_inverse_*` families correspond to the current frozen-LM comparison. Older `*_inverse_per_case_results.csv` files are retained as historical source data; their names must not be used to infer that they contain the current optimizer.

The competition representative forward curve is explicitly named `competition_forward_recovered_from_svg.csv`: the original plotting script recovered it from an archived vector figure. It is suitable for redrawing that illustration, not for recomputing full-precision numerical error metrics.

The grid manuscript baseline is refitted from 1,024 training parameter points, each with 24 binary outcomes. `scripts/refit_grid_gbt.py` uses the saved training-only tuning choices and reproduces the 64 archived predictions. `entry grid-gbt` runs the older application-grid cross-validation audit; that audit is a different protocol.
