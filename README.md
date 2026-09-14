# Aerodynamic learning

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
| What do $C_D$, $C_L$, and $C_M$ look like over an $(\alpha, Re)$ domain? | [`notebooks/simple_analysis.ipynb`](./notebooks/simple_analysis.ipynb) | NeuralFoil polar, angle sweep, two-dimensional coefficient fields, and a simple timing experiment for a symmetric NACA airfoil | Implemented exploratory baseline |
| How different are NeuralFoil and XFoil over the operating domain? | [`notebooks/nf_v_xfoil.ipynb`](./notebooks/nf_v_xfoil.ipynb) | One-dimensional polar comparison and $(\alpha, Re)$ residual fields for $C_D$ and $C_L$, including missing XFoil convergence points | Experimental; stored outputs include environment errors and should be rerun before using the numerical observations |
| How does a NeuralFoil-driven multipoint shape optimization change an airfoil and its fields? | [`notebooks/optimisation.ipynb`](./notebooks/optimisation.ipynb) | CST/Kulfan shape optimization across Reynolds numbers, followed by baseline/optimized geometry, polar, sweep, and field comparisons | Implemented proof of concept; not a validated design study |
| How can a parameterized NACA family be organized by whole aerodynamic response surfaces? | [`notebooks/multivariate_pca_clustering.ipynb`](./notebooks/multivariate_pca_clustering.ipynb) | Database generation and mean-$C_D$/mean-$C_L$ maps over geometric parameters | In progress: multivariate functional PCA and clustering are not yet implemented |
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
datasets using NeuralFoil or XFoil. The clustering notebook currently uses this
workflow to compare operating-domain averages against camber and thickness.
The intended next step is to retain the complete surfaces, construct a
multivariate functional representation, and assess clusters in a sufficiently
rich score space before displaying them in two dimensions.

Focused tests for the database layout, NeuralFoil sampling, and representation
of unconverged XFoil points are in
[`tests/test_generate_airfoil_database.py`](./tests/test_generate_airfoil_database.py).

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

Near-term priorities are to complete and validate the functional PCA/clustering
experiment, establish repeatable NeuralFoil/XFoil benchmarks over multiple
airfoils, and implement a simple Gaussian-process calibration baseline before
considering neural operators or broader optimization studies.

## Reproducing the notebooks

Install the pinned Python dependencies:

```bash
python -m pip install -r requirements.txt
```

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
