# Two-dimensional quantity-aware FDM: execution evidence

Report: `../report/qoi_fdm.pdf` (12 pages). Original experiments are preserved.

## Question and design

Compare 12 scalar objectives using the existing quantity-aware functional diffusion map with exactly two coordinates. Assess conditional QoI preservation, whole-field neighborhoods, shape recovery, decoded objective errors, round-trip consistency, and actual latent-optimization gains.

The original 768-shape CST cache is reused. Confidence screening retains 598; requiring all nine prescribed-lift crossings retains a common cohort of 554. Seeds 42/43/44 yield 360/83/111 train/validation/test splits. Only validation selects kernel and objective-surrogate parameters. The encoder consumes true fields and objective values, so this is not blind aerodynamic prediction.

## Reproduction

With the project's analysis dependencies available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/qoi-mpl python notebooks/qoi_fdm.py
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/qoi-mpl python notebooks/qoi_fdm_report.py
python -m unittest discover -s tests -v
cd report
pdflatex -interaction=nonstopmode -halt-on-error qoi_fdm.tex
bibtex qoi_fdm
pdflatex -interaction=nonstopmode -halt-on-error qoi_fdm.tex
pdflatex -interaction=nonstopmode -halt-on-error qoi_fdm.tex
```

The actual run used `env/bin/python` with `PYTHONPATH=/home/rraanto/.local/lib/python3.13/site-packages` to access existing scikit-learn. No dependencies were installed. Versions, source hashes, and raw cache hash are in `data/qoi_fdm/metadata.json`. The script is canonical; the companion notebook invokes it.

## Evidence

- 432 kernel fits; 36 selected maps; 360 decoded holdout evaluations; 96 candidate designs, each checked with coarse/fine grids and a larger network.
- Mean conditional NRMSE: 0.013–0.116 across objectives; mean rank correlation exceeds 0.9997 for all objectives.
- Eleven of twelve mean field errors exceed the training-mean baseline. Mean shape predictions beat latent nearest-neighbor shape predictions on the same test populations.
- Hull coverage averages 76.9–88.6%; supported barycentric shape RMS is 0.897–1.040% chord.
- Decoded objective errors are much larger than conditional objective errors. Minimum confidence remains above .75 in the decoded holdout sample.
- Mean-efficiency candidate improves 0.680% on the coarse grid, 0.678% on the refined grid, and 1.160% with `xxxlarge`; round-trip residual is 0.455 median latent distances.
- All other objectives return negligible gains or deterioration. Retain the incumbent for worse candidates. These are NeuralFoil results, not independently validated physical gains.
- Exact differentiable inversion on the open 15-dimensional family cannot be supplied by two coordinates. A representative decoder can still be useful for restricted design, but consistency and actual performance must be checked separately.

## Validation

All 24 repository tests passed, including two new analytic objective/trim tests. LaTeX compiled with resolved citations and no overfull-box warnings. All 12 rendered pages were visually checked; detailed tables and embedding panels were inspected separately. Raw input data and previous report sources were not modified.
