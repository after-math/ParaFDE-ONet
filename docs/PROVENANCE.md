# Provenance

The release was assembled on 2026-09-27 from the supplied article workspace and `trial` directory. The selected manuscript is the shortened version titled “ParaFDEONet: A Unified Framework for Forward Prediction and Parameter Identification in Parameterized Functional Differential Equations”. The manuscript itself is not included in this code package.

`source_projects.json` maps the original experiment directory names to the new public layout. `source_manifest.json` records each copied file's relative source path, original SHA-256, release SHA-256 and whether it changed. Source roots are aliases, so the public record does not expose an author's absolute workstation path.

## Selection

The release includes the three main synthetic benchmarks, their paired sensitivity variants, the grid application and the complete CSTR LM implementation. Shared-panel inverse solvers, derivative checks, timing benchmarks and the historical supplementary inverse implementations are included. Unrelated exploratory systems, manuscript drafts, remote-launch scripts, environment dumps, caches, logs, and large neural-network checkpoints are excluded.

The component/history/parameter branch implementations remain in their original project modules. The competition sensitivity extension reads its parent implementation with checked patch anchors; that parent source is included. This release does not replace the trained architectures with a newly designed unified implementation.

## Changes

- Added a repository-level launcher, dependency list, documentation, citation metadata, isolated test runner, source checks and artifact preparation.
- Converted cross-project references to repository-relative paths and documented the separate frozen-LM entry point.
- Disabled tracking and notifications and removed the author-specific notification endpoint.
- Applied the requested SciencePlots theme and Source Han Serif font to packaged plotting code.
- Adapted manuscript plotting scripts to bundled data and separate output locations.
- Added a GBT fixed-configuration refit utility using archived training labels and saved tuning choices. Its predictions were checked against the original 64 application values.
- Kept historical random seeds, scientific configurations, model-format names, solver formulas, losses and inverse algorithms.

Some copied results are reformatted JSON with sanitized paths; the original and release hashes distinguish these from unchanged numerical data. The new release utilities do not claim to be historical experimental source files.

No license text was present in the selected source projects. No new license, publication DOI, model-download URL or unverified publication status has been assigned.
