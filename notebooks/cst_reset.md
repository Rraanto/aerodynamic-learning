# CST research reset: execution evidence

Question: Can a reduction of aerodynamic fields into two or three coordinates
preserve a chosen quantity and provide a reliable map back to feasible shapes
that makes performance optimization useful?

Specification: `../next_experimentations.md`. The active corpus is now directly
sampled standard CST, with 15 independent shape variables. Previous NACA work
is historical and its data are not mixed into the experiment.

Reproduce with `python notebooks/cst_reset.py` from the root, or execute
`cst_reset.ipynb`. The script builds a hash-keyed cache if required, then
generates all evidence, report tables, and seven figure sets. Build the PDF
from `report/` using `pdflatex`, `bibtex`, and two more `pdflatex` passes.

## Setup

- 768 directly generated CST shapes; degree seven, class exponents 0.5 and 1;
  fixed trailing-edge gap 0.0025 and no leading-edge modification.
- 31 × 21 angle/Reynolds grid, [−5,12] degrees × [0.3,3] million. Reynolds
  sampling is geometric and functional integration is uniform in log Reynolds.
- CD/CL/CM fields and pointwise NeuralFoil confidence, model `large`, Mach zero,
  natural transition with `n_crit=9`.
- Primary cohort: 598 shapes with confidence ≥0.75 throughout the domain;
  all 170 excluded samples remain in the raw database. Splits: 388/89/121.
- Baseline: weighted multivariate FPCA. Nonlinear control: ordinary functional
  diffusion maps. Target method: functional kernel augmented with normalized
  squared QoI differences, tuned on validation only. Target default: minimum
  sampled-domain CD. Alternative quantities and similarity are CLI switches.
- Every encoder consumes complete fields; the quantity-aware extension computes
  the QoI from those fields. Holdout results are conditional preservation tests,
  not blind geometry-to-performance predictions.

## Evidence and interpretation

- FPCA retains 68.1%/77.2% variance in two/three PCs; 26 PCs are needed for 99%.
  Direct linear test field errors are 0.287/0.206.
- Quantity-aware two/three coordinates give conditional scalar NRMSE
  0.0047/0.0054 and near-perfect rank ordering. Since the scalar is explicitly
  included in the similarity and extension, this does not establish an
  independent predictive capability; `z=q` is an exact trivial scalar control.
- Whole-field neighbor errors increase to 1.264/1.250: worse than predicting
  the training mean under the chosen field metric. Five-neighbor field recall
  falls to 3.3%; FPCA-3 achieves 43.3%.
- The target-aware map mainly orders the scalar and does not separate contour
  or field differences within equal-scalar neighborhoods. Thickness and camber
  coloring, cluster memberships and neighbor diagnostics expose this loss.
- Ordinary FDM-2 localizes a spectral coordinate on a few points; cluster counts
  are 588/1/8/1. Broader bandwidths were checked and the pilot evidence remains
  saved. FDM-3 selects bandwidth 4 with more distributed coordinates. This is
  a kernel/data diagnostic, not a general ranking of nonlinear methods.
- Eight-start hull-constrained RBF-surrogate optimization decodes each candidate
  through barycentric CST interpolation. All reported candidates pass geometry
  and confidence checks. Optimizer status is saved separately from those checks.
- Target-aware candidates improve the `large` model's grid minimum by ~2.6%,
  with round-trip errors 0.16–0.27 typical latent pair distances. The hull is
  not established to be the true attainable field-embedding image.
- Finer-grid checks reduce the gain to ~1.4%. A solver-size check reverses the
  target-aware gain: candidates are ~1.1–1.2% worse than the incumbent under
  `xxxlarge`. This is sensitivity within the same surrogate family, not
  validation against higher-fidelity physics.
- FPCA and ordinary FDM-3 candidates fail to improve the incumbent. The adopted
  design file protects against degrading the known best training shape.
- 87.1% of grid minima occur at the largest Reynolds number. This objective
  lacks a lift requirement and is not a measure of whole-domain performance.

## Scope and next experiment

Observed scalar preservation is strong; useful full-field invertibility and
robust optimization are not established. Standard rounded-nose CST ordinates
are smooth on the open chord interval but are not C1 graphs at x=0. A minimum
over operating points can switch active points and be nonsmooth. Neither issue
is resolved by fitting a smooth latent surrogate.

Next test: joint QoI/field/round-trip constraints on reduction and decoding,
a prescribed-lift mean-drag objective, and independent aerodynamic validation.
Confidence conditioning, one Sobol realization and one split limit the result.

The code creates no new dependency or parallel research layout. Reusable
methods are in `utils/cst_reduction.py`; orchestration remains with this
experiment. Dataset schema and provenance are in `data/cst_reset/README.md`.
Every publication figure comes from the script and persisted arrays/tables.
