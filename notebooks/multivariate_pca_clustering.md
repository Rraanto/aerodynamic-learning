# Multivariate functional PCA: execution evidence

- Population: 170 airfoils; grid: 50 × 50; cache: `fields_f172ec2047e982df.npz`.
- Full-population 99% variance rank: 7; two PCs retain 88.29%.
- Held-out reconstruction at training-selected rank 7: relative field SSE 0.0112, CD RMSE 0.00247, CL RMSE 0.02526.
- K-means, k=4: relative field SSE MFPCA=0.2612, camber=0.6131, full geometry=0.6722; functional/camber ARI=0.329; MFPCA subsample ARI mean=0.889, min=0.459.
- Ward, k=4: relative field SSE MFPCA=0.2782, camber=0.6715, full geometry=0.5938; functional/camber ARI=0.125; MFPCA subsample ARI mean=0.637, min=0.440.
- GMM, k=4: relative field SSE MFPCA=0.3682, camber=0.7376, full geometry=0.7152; functional/camber ARI=0.103; MFPCA subsample ARI mean=0.581, min=0.421.
- K-means representation check: 2-PC SSE=0.2623, means SSE=0.3090, unreduced fields SSE=0.2612. Two PCs perform nearly as well for this partition despite retaining only 88% of variance; richer scores matter more for reconstruction.
- Sensitivity ARI versus primary partition, balanced-0.95: K-means=0.953, Ward=0.561, GMM=0.401.
- Sensitivity ARI versus primary partition, balanced-0.999: K-means=1.000, Ward=0.760, GMM=0.676.
- Sensitivity ARI versus primary partition, unbalanced-0.99: K-means=0.272, Ward=0.380, GMM=0.385.
- Sensitivity ARI versus primary partition, coarse-grid-0.99: K-means=1.000, Ward=0.460, GMM=1.000.
- Nonconverged fits: main=0, subsampling=0, sensitivity=0.
- Interpretation: field SSE measures compression under the chosen balanced metric; functional clustering is designed for this metric. ARI with geometry is agreement, not ground-truth accuracy.
- Limitations: no physical class labels, confidence filtering, CFD validation, or extrapolation test. Symmetric-airfoil position is a convention; consult no-symmetric sensitivity rows. Coarse-grid checks are not solver convergence.
- Figure content: reduction = variance and held-out error; sweeps = matched-k common-metric comparisons; maps = local cluster colors and explicit thickness offsets in camber geometry; surfaces = cluster means with shared coefficient scales; polar bands = member 10–90% spread.
- Reproduce: execute all cells of `notebooks/multivariate_pca_clustering.ipynb`; numeric inputs are reused only when the generation configuration matches. Tables and fitted PCA arrays are in `data/multivariate_pca_clustering/`; PDFs/PNGs are in `figures/multivariate_pca_clustering/`.
