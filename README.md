# Aerodynamic learning

The latest focused study compares **12 objectives using two-dimensional quantity-aware FDM**, with decoded shape and optimization checks. See the [PDF report](report/qoi_fdm.pdf), [execution evidence](notebooks/qoi_fdm.md), and [notebook](notebooks/qoi_fdm.ipynb). Reproduce with `python notebooks/qoi_fdm.py` followed by `python notebooks/qoi_fdm_report.py`.

The active research has reset to a **directly sampled Class--Shape
Transformation (CST) dataset**, following
[`next_experimentations.md`](./next_experimentations.md). Start with the
[CST experiment notebook](./notebooks/cst_reset.ipynb),
[execution evidence](./notebooks/cst_reset.md), and
[PDF report](./report/cst_reset.pdf)
([LaTeX source](./report/cst_reset.tex)).
The [dataset documentation](./data/cst_reset/README.md) describes 768 shapes
with drag, lift, moment, and pointwise NeuralFoil confidence. The earlier NACA
experiments below remain historical baselines.

The new experiment compares functional PCA with ordinary and quantity-aware
functional diffusion maps in two and three dimensions, then verifies decoded
optimization candidates. Preserving minimum drag closely does not preserve
whole-field structure, and the small apparent optimization gain reverses
with a different NeuralFoil model size. These are exploratory surrogate results.

Run from the repository root with the project's numerical analysis dependencies:

```bash
python notebooks/cst_reset.py
python notebooks/cst_reset.py --qoi mean_drag
python notebooks/cst_reset.py --metric drag
python -m unittest discover -s tests -v
```

The first command creates or reuses the new generation cache; alternative
objectives and metrics write separate results under `data/cst_reset/` and
`figures/cst_reset/`. Actual library versions and source/cache hashes are recorded
with each run. Build the default report from `report/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error cst_reset.tex
bibtex cst_reset
pdflatex -interaction=nonstopmode -halt-on-error cst_reset.tex
pdflatex -interaction=nonstopmode -halt-on-error cst_reset.tex
```

This repository explores airfoil aerodynamics as **functions over an operating
domain**, rather than only as values at one design point. For an airfoil
described by geometry parameters $K$, the long-term objective is to learn and
use fields such as

$$
(K, \alpha, Re) \longmapsto \bigl(C_D, C_L, C_M\bigr),
$$

where $\alpha$ is the angle of attack and $Re$ is the Reynolds number. The
current work is exploratory: it establishes field-generation and comparison
workflows, tests a first multipoint optimization, and prepares the data and
literature basis for functional dimension reduction, clustering, and
multi-fidelity learning.

The project rationale and original roadmap are recorded in
[`desc.ipynb`](./desc.ipynb).

## Which notebook should I use?

| Research question | Notebook | What it currently contains | Maturity |
| --- | --- | --- | --- |
| Can quantity-aware functional coordinates preserve an objective and support feasible shape optimization? | [`notebooks/cst_reset.ipynb`](./notebooks/cst_reset.ipynb) | New 15-variable CST corpus; 2D/3D FPCA and diffusion maps; conditional quantity/field preservation; CST decoding, optimization, and grid/model-size sensitivity | Active research reset; scalar preservation succeeds while field recovery and robust optimization remain unresolved |
| What do $C_D$, $C_L$, and $C_M$ look like over an $(\alpha, Re)$ domain? | [`notebooks/simple_analysis.ipynb`](./notebooks/simple_analysis.ipynb) | NeuralFoil polar, angle sweep, two-dimensional coefficient fields, and a simple timing experiment for a symmetric NACA airfoil | Implemented exploratory baseline |
| How different are NeuralFoil and XFoil over the operating domain? | [`notebooks/nf_v_xfoil.ipynb`](./notebooks/nf_v_xfoil.ipynb) | One-dimensional polar comparison and $(\alpha, Re)$ residual fields for $C_D$ and $C_L$, including missing XFoil convergence points | Experimental; stored outputs include environment errors and should be rerun before using the numerical observations |
| How does a NeuralFoil-driven multipoint shape optimization change an airfoil and its fields? | [`notebooks/optimisation.ipynb`](./notebooks/optimisation.ipynb) | CST/Kulfan shape optimization across Reynolds numbers, followed by baseline/optimized geometry, polar, sweep, and field comparisons | Implemented proof of concept; not a validated design study |
| How can a parameterized NACA family be organized by whole aerodynamic response surfaces? | [`notebooks/multivariate_pca_clustering.ipynb`](./notebooks/multivariate_pca_clustering.ipynb) | Weighted multivariate functional PCA; K-means, Ward, and Gaussian mixtures compared with camber, full geometry, and coefficient means; reconstruction and stability checks | Reproducible NeuralFoil experiment; see [execution evidence](./notebooks/multivariate_pca_clustering.md) |
| How can low- and high-fidelity simulations be combined or calibrated? | [`notebooks/model_calibration.ipynb`](./notebooks/model_calibration.ipynb) | Problem statement only | Planned; start with the [multi-fidelity literature map](./reviews/multi-fidelity/review.md) |

Use [`notebooks/simple_analysis.ipynb`](./notebooks/simple_analysis.ipynb) as
the entry point for the numerical conventions and field representation. The
other notebooks then branch into fidelity assessment, optimization, and
population-level analysis.

## Current scientific content

### Aerodynamic response fields

The baseline notebook evaluates a NACA 0006 airfoil with NeuralFoil on angle-of-attack and Reynolds-number grids. It illustrates expected symmetry in the drag and lift fields of a symmetric airfoil and shows how stall-like transitions and Reynolds-number dependence appear in the surrogate response. These are qualitative, model-based observations with an immediate perspective of CFD-based validation 

### Fidelity comparison

The NeuralFoil/XFoil notebook aligns converged XFoil points with a regular NeuralFoil grid and visualizes pointwise discrepancies. It is the appropriate starting point for defining a low-/higher-fidelity correction problem. The comparison is presently limited to one airfoil and a narrow operating regime; the checked-in execution also contains dependency errors, so its quoted error magnitudes should be treated as provisional.

### Multipoint shape optimization

The optimization workflow uses 16 Kulfan/CST weights and independently trimmed angles of attack at selected Reynolds numbers. The current objective minimizes mean NeuralFoil drag subject to geometric validity, model-confidence, and non-negative-lift constraints. It demonstrates the mechanics of optimizing a field-aware design and inspecting gains and losses across the operating domain; it does not yet establish robustness, uniqueness, or transfer to a higher-fidelity model.

### Airfoil populations and functional representations

[`utils/generate_airfoil_database.py`](./utils/generate_airfoil_database.py)
generates parameterized NACA four-digit families and long-form $C_D$/$C_L$
datasets using NeuralFoil or XFoil. The clustering notebook compares complete
surfaces for 170 NACA airfoils on a 50-by-50 operating grid. It balances lift and
drag by integrated variance, retains 99% of functional variance, and compares
three clustering algorithms with camber-only and full-geometry baselines using
a shared aerodynamic metric. Held-out reconstruction, matched cluster-count
sweeps, refitted subsampling, and scaling/grid sensitivity checks accompany
the two-dimensional displays. Numeric caches and result tables are saved in
`data/multivariate_pca_clustering/`, with figures in
`figures/multivariate_pca_clustering/`.

Focused tests for the database layout, NeuralFoil sampling, and representation
of unconverged XFoil points are in
[`tests/test_generate_airfoil_database.py`](./tests/test_generate_airfoil_database.py).

The [geometry and regime extension](./notebooks/airfoil_regimes.ipynb), with
[execution notes](./notebooks/airfoil_regimes.md), measures
actual contour-neighborhood preservation on the saved 2D points, partitions
angle magnitude and Reynolds number into nine operating bins, and compares
whole-field, regime-performance, and regime-local clusters. Its
[minimal LaTeX report](./report/airfoil_regimes.tex) records performance
tradeoffs, reconstruction diagnostics, and sensitivity to bin boundaries and
incidence sign. The [compiled report](./report/airfoil_regimes.pdf) includes
the resulting figures. These groups are exploratory surrogate-based results.

Regenerate this extension from the root with the numerical analysis dependencies:

```bash
python notebooks/airfoil_regimes.py
```

This reuses the original aerodynamic cache and saves new evidence under
`data/airfoil_regimes/` and `figures/airfoil_regimes/`. Actual software versions
and source hashes are recorded in its `run_metadata.json`; exact assignments
may vary across numerical library versions. Build the report from `report/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error airfoil_regimes.tex
bibtex airfoil_regimes
pdflatex -interaction=nonstopmode -halt-on-error airfoil_regimes.tex
pdflatex -interaction=nonstopmode -halt-on-error airfoil_regimes.tex
```

## Literature maps

- [`reviews/field-learning/review.md`](./reviews/field-learning/review.md)
  surveys multivariate functional PCA, cluster-aware subspaces,
  tensor-product representations, nonlinear embeddings, and aerodynamic-field
  latent representations.
- [`reviews/multi-fidelity/review.md`](./reviews/multi-fidelity/review.md)
  surveys autoregressive Gaussian processes/co-kriging, reduced-basis field
  fusion, spatio-temporal Gaussian processes, neural operators, and discrepancy
  calibration.

Each review has a companion `references.bib` and a `query.yaml` that records its
scope and selection priorities.

### AI-assistance disclosure

The literature searches, bibliography preparation, and synthesis in
`reviews/` were performed with AI assistance. The reviews are navigation aids,
not authoritative or systematic reviews. Researchers should read and verify
the cited primary sources before relying on a claim, method, or bibliographic
record. Numerical notebook results likewise remain the responsibility of the
researcher who reruns and validates the corresponding experiment.

## Intended perspectives

The planned research progression is:

1. define representative airfoil families, operating domains, and fidelity
   levels;
2. quantify variation and numerical uncertainty in complete aerodynamic
   response fields;
3. compare functional bases, nonlinear embeddings, and clustering methods using
   reconstruction quality, stability, and aerodynamic interpretability;
4. learn geometry-to-field or geometry-to-latent mappings, with explicit
   low-/high-fidelity calibration and uncertainty where useful;
5. use learned functional descriptors in robust, multipoint optimization; and
6. test transfer from NeuralFoil/XFoil screening to higher-fidelity CFD and,
   eventually, experimental data.

Near-term priorities are to validate the functional PCA/clustering results
against higher-fidelity models, establish repeatable NeuralFoil/XFoil benchmarks over multiple
airfoils, and implement a simple Gaussian-process calibration baseline before
considering neural operators or broader optimization studies.

## Reproducing the notebooks

Install the pinned Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The functional PCA notebook additionally uses the standard clustering estimators
and notebook tooling in `requirements-analysis.txt`:

```bash
python -m pip install -r requirements-analysis.txt
python -m unittest discover -s tests -v
```

Select the installed environment as the notebook kernel and run
all cells. This notebook works from the root or `notebooks/`; it reuses only
generation caches matching its recorded configuration. Its generated result
note records numerical evidence and limitations.

XFoil analyses additionally require an XFoil executable available on `PATH`.
[`utils/run_xfoil_patch.py`](./utils/run_xfoil_patch.py) applies a local parser
compatibility patch for the AeroSandbox version used here. Run notebooks from
the `notebooks/` directory because several of them use relative `%run` paths.

The repository currently records exploratory notebook outputs rather than a
fully automated end-to-end reproduction pipeline. Before citing a quantitative
result, rerun the relevant notebook in a clean environment, inspect convergence
and missing values, and record the software and solver versions used.

Additional vendored AeroSandbox documentation is indexed in
[`docs/Readme.md`](./docs/Readme.md).
