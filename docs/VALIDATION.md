# Validation

Local release checks were performed on 2026-09-27 using macOS, Python 3.12.7 and CPU execution. The exact installed scientific package versions are recorded in `requirements-tested.txt`.

## Tests

| Suite | Passed |
| --- | ---: |
| Competition with sensitivity | 5 |
| Competition without sensitivity | 10 |
| SEI with sensitivity | 9 |
| SEI without sensitivity | 9 |
| Nicholson with sensitivity | 6 |
| Grid forward, screening and GBT diagnostics | 23 |
| CSTR forward and online workflow | 11 |
| Shared-panel inverse and frozen LM | 13 |
| Sensitivity Jacobian | 6 |
| Feature reuse | 5 |
| **Total** | **97** |

Each original suite runs in a separate process because the research projects use shared module names such as `model`, `data`, and `equation`. PyTorch emitted deprecation warnings for the inherited TorchScript solver code; those warnings did not fail the tests.

## Execution

- The five forward smoke pipelines completed data generation, small-network training, checkpoint saving and evaluation on CPU.
- Competition, SEI, Nicholson and grid recorded `smoke_complete` with `success=true`.
- CSTR recorded `status=completed` after three iterations, with `passed=false` for the retained formal accuracy gates. This expected nonconvergence is recorded explicitly; the smoke run is an execution check, not an accuracy result.
- A reduced competition shared-panel run completed both direct LM and direct DE inversion, then aggregated its panel results.
- All 14 advanced entry points returned their command-line help successfully.
- The CSTR archive verifier checked 252 inversion records, 20 event files and the archived operating decisions.
- The manuscript GBT was refitted from archived training labels. Its primary 64 predictions matched exactly; all five recorded refits agreed to floating-point precision. The primary MAE was 2.7979426773 percentage points.
- Seven figure outputs were regenerated from bundled source data: forward examples, inverse examples, sensitivity ratios, grid survivability, CSTR workflow, feature reuse and PINN-DDE feasibility. Representative forward, grid and CSTR plots were visually inspected.

## Scope

The full-size networks were not retrained, and the full GPU inverse/timing studies were not rerun during packaging. Archived paper metrics remain identified as original experimental outputs. CPU smoke checks do not validate the paper's reported accuracy or hardware speedups.

Run `python run.py check` to validate source syntax, JSON files, cross-project source paths, release hashes and accidental workstation-specific settings. Run `python run.py test all` for the original test suites.
