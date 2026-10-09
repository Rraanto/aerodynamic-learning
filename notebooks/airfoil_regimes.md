# Geometry and operating-regime analysis

Question: Does proximity in the existing functional 2D map preserve actual
contour similarity, and do whole-field clusters retain their meaning across
low/middle/high operating regimes?

Reproduce from the root with `python notebooks/airfoil_regimes.py` in an
environment providing NumPy, SciPy, pandas, Matplotlib, and scikit-learn.
The script uses the existing cache and saved PCA coordinates and partitions;
it does not rerun NeuralFoil or overwrite the earlier experiment.

Outputs are in `data/airfoil_regimes/`, `figures/airfoil_regimes/`, and
`report/airfoil_regimes.tex`. Build instructions are in the report and README.
`run_metadata.json` records the actual environment and hashes. Existing global
partitions came from scikit-learn 1.7.2; new partitions use the version recorded
in this run. Exact cluster membership can depend on numerical library versions.

- Geometry metric: registered normal-offset NACA contours, 201 cosine-spaced
  stations, trapezoidal chordwise integration, finite trailing edge. It avoids
  interpreting the meaningless camber position of symmetric airfoils as a
  shape difference. Shape parameter correlations still use the recorded zero
  position convention.
- Five-neighbor recovery is 36.2% in 2D, 44.7% in seven PCs, and 45.3% in the
  full fields; chance expectation is 3.0%. Pairwise correlation is descriptive,
  since pairs are dependent. Counterexamples are selected by a fixed 5%
  distance-tail rule, rather than chosen manually.
- Nine regimes cross angle magnitude [0,5], [5,10], [10,15] degrees with
  equal-width linear Reynolds intervals over [1,2] million. These are bins,
  not a diagnosis of stall. Nodal-hat quadrature clips exact bin boundaries;
  weighted regime volumes reconstruct the original quadrature.
- Performance includes mean drag, signed lift, absolute lift, and mean
  pointwise absolute-lift/drag. Clustering uses 27 standardized drag,
  absolute-lift and efficiency summaries; signed lift is retained for diagnosis.
  Equal metric weights are a declared choice, not learned mission utilities.
- Regime-performance clusters have sizes 55, 54, 27, 34, silhouette 0.323,
  and ARI 0.488 against whole-field seven-PC clustering. R4 has the best mean
  low-angle efficiency; R1 has the best middle- and high-angle efficiency.
  Member quantiles are population spread, not uncertainty bounds on means.
- Two global PCs lose 3.2–3.4% of low-angle field energy and 11.8–17.7% of
  high-angle energy. Seven PCs keep regime reconstruction errors below 1.6%.
  These diagnostics fit and evaluate the same population. The previous
  notebook, not this extension, provides the held-out reconstruction test.
- Local field clustering and global partitions share fixed original
  coefficient scales. Local clustering is expected to reduce its own SSE;
  it is not a held-out superiority claim. Local colors are not matched across
  regime panels.
- Robustness is incomplete: shifted angle boundaries yield ARI 0.460/0.515;
  positive/negative incidence yield 0.381/0.500. Subsample stability averages
  0.899 but has minimum 0.460. This is not evidence of physical classes.
- Limitation: surrogate-only data, no retained confidence field, no fixed-lift
  comparison or trim, and no higher-fidelity validation. Next check: assess a
  positive-incidence prescribed-lift mission with confidence filtering and
  representative higher-fidelity calculations.

Figure provenance: `notebooks/airfoil_regimes.py` orchestrates all six figures;
`utils/airfoil_regimes.py` implements numerical methods. `summary.json`,
`points.csv`, `pair_distances.csv`, `performance.csv`, `cluster_performance.csv`,
`cluster_metrics.csv`, and `regimes.csv` contain the corresponding evidence.

Sources verified for the report: Happ and Greven's arXiv v3 metadata and
abstract, NASA NTRS NACA-TR-460 record, and the NeuralFoil project documentation.
Contour equations were also checked against the installed AeroSandbox family
generator. The regime definitions and screening objectives are this experiment's
choices, rather than claims attributed to these references.
