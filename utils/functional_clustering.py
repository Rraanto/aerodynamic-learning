"""Quadrature-weighted PCA and diagnostics for complete multivariate fields."""

import numpy as np
from scipy.linalg import svd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def trapezoid_weights(grid):
    """Normalized integration weights on a finite, strictly increasing grid."""
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all():
        raise ValueError("Grid must contain at least two finite coordinates.")
    steps = np.diff(grid)
    if np.any(steps <= 0):
        raise ValueError("Grid must be strictly increasing.")
    weights = np.r_[steps[0], steps[:-1] + steps[1:], steps[-1]] / 2
    return weights / (grid[-1] - grid[0])


class FunctionalPCA:
    """Dense-grid MFPCA in a product L2 metric, with optional output balancing.

    Fields have shape (sample, alpha, Reynolds, coefficient). Each coefficient
    is scaled by its training integrated standard deviation, never pointwise.
    """

    def __init__(self, alphas, reynolds, balance=True):
        self.weights = np.outer(trapezoid_weights(alphas), trapezoid_weights(reynolds))
        self.balance = balance

    def _validate(self, fields):
        fields = np.asarray(fields, dtype=float)
        if fields.ndim != 4 or fields.shape[1:3] != self.weights.shape:
            raise ValueError("Fields must have shape (sample, alpha, Re, coefficient).")
        if not np.isfinite(fields).all():
            raise ValueError("Fields contain missing or non-finite values; no imputation is applied.")
        return fields

    def fit(self, fields):
        """Estimate mean, coefficient scales, and orthogonal weighted modes."""
        fields = self._validate(fields)
        if len(fields) < 2:
            raise ValueError("At least two fields are required.")
        self.mean_ = fields.mean(axis=0)
        variance = np.sum(fields.var(axis=0, ddof=1) * self.weights[..., None], axis=(0, 1))
        self.scale_ = np.sqrt(variance) if self.balance else np.ones(fields.shape[-1])
        if np.any(self.scale_ <= 0):
            raise ValueError("Each balanced coefficient must have positive integrated variance.")
        self.factor_ = np.sqrt(self.weights[..., None]) / self.scale_
        matrix = self.weighted(fields)
        _, singular, self.components_ = svd(matrix, full_matrices=False)
        if singular[0] == 0:
            raise ValueError("PCA requires nonzero field variance.")
        self.explained_variance_ = singular**2 / (len(fields) - 1)
        self.explained_variance_ratio_ = singular**2 / np.sum(singular**2)
        return self

    def weighted(self, fields):
        """Map fields into the centered Euclidean form of the fitted L2 metric."""
        fields = self._validate(fields)
        if fields.shape[1:] != self.mean_.shape:
            raise ValueError("Coefficient count differs from fitted fields.")
        return ((fields - self.mean_) * self.factor_).reshape(len(fields), -1)

    def rank(self, fraction=0.99):
        """Smallest rank retaining the requested training variance fraction."""
        if not 0 < fraction <= 1:
            raise ValueError("Variance fraction must lie in (0, 1].")
        return min(len(self.components_), int(np.searchsorted(
            np.cumsum(self.explained_variance_ratio_), fraction
        )) + 1)

    def transform(self, fields, rank):
        """Project fields onto the first rank modes without whitening scores."""
        if not 1 <= rank <= len(self.components_):
            raise ValueError("Rank is outside the fitted component range.")
        return self.weighted(fields) @ self.components_[:rank].T

    def inverse_transform(self, scores):
        """Reconstruct coefficient fields in their original units."""
        scores = np.asarray(scores)
        flat = scores @ self.components_[:scores.shape[1]]
        return flat.reshape((-1,) + self.mean_.shape) / self.factor_ + self.mean_


def representations(fields, geometry, alphas, reynolds, fraction=0.99):
    """Fit the functional and standardized geometric/mean-field baselines."""
    model = FunctionalPCA(alphas, reynolds).fit(fields)
    means = np.sum(fields * model.weights[None, ..., None], axis=(1, 2))
    return model, {
        "MFPCA": model.transform(fields, model.rank(fraction)),
        "MFPCA-2D": model.transform(fields, 2),
        "Field means": StandardScaler().fit_transform(means),
        "Camber (m,p)": StandardScaler().fit_transform(geometry[:, :2]),
        "Geometry (m,p,t)": StandardScaler().fit_transform(geometry),
    }


def fit_clusters(features, method, k, seed=42):
    """Fit a reproducible partition and expose mixture convergence/uncertainty."""
    info = {"converged": True, "bic": np.nan, "mean_max_probability": np.nan}
    if method == "K-means":
        model = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=seed)
    elif method == "Ward":
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    elif method == "GMM":
        model = GaussianMixture(n_components=k, covariance_type="full", n_init=5,
                                max_iter=500, reg_covar=1e-6, random_state=seed)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    labels = model.fit_predict(features)
    if method == "GMM":
        info.update(converged=bool(model.converged_), bic=float(model.bic(features)),
                    mean_max_probability=float(model.predict_proba(features).max(axis=1).mean()))
    return labels, info


def field_metrics(fields, labels, metric):
    """Evaluate every partition in the same full-field metric and coefficient units."""
    groups, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    centroids = np.stack([fields[labels == group].mean(axis=0) for group in groups])
    reconstructed = centroids[inverse]
    residual = fields - reconstructed
    sse = np.sum((residual * metric.factor_)**2)
    total = np.sum(((fields - fields.mean(axis=0)) * metric.factor_)**2)
    rmse = np.sqrt(np.mean(np.sum(residual**2 * metric.weights[None, ..., None], axis=(1, 2)), axis=0))
    return {
        "field_silhouette": silhouette_score(metric.weighted(fields), labels)
        if 1 < len(groups) < len(fields) else np.nan,
        "relative_field_sse": sse / total,
        "CD_rmse": rmse[0], "CL_rmse": rmse[1],
        "min_cluster_size": counts.min(), "max_cluster_size": counts.max(),
        "actual_k": len(groups),
    }


def reconstruction_metrics(fields, reconstructed, metric):
    """Integrated reconstruction RMSE and error relative to training-mean prediction."""
    residual = fields - reconstructed
    rmse = np.sqrt(np.mean(np.sum(residual**2 * metric.weights[None, ..., None], axis=(1, 2)), axis=0))
    return {"CD_rmse": rmse[0], "CL_rmse": rmse[1],
            "relative_field_sse": np.sum((residual * metric.factor_)**2)
            / np.sum(metric.weighted(fields)**2)}


def plot_reduction(model, reconstruction):
    """Plot variance retention and held-out field reconstruction error."""
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    cumulative = np.cumsum(model.explained_variance_ratio_)
    end = min(20, len(cumulative))
    axes[0].plot(np.arange(1, end + 1), cumulative[:end], "o-")
    axes[0].axhline(0.99, color="0.4", ls="--", label="99%")
    axes[0].set(xlabel="Retained components", ylabel="Cumulative variance", ylim=(0, 1.02))
    axes[0].legend()
    for split, group in reconstruction.groupby("split", sort=False):
        axes[1].semilogy(group["rank"], group["relative_field_sse"], "o-", label=split)
    axes[1].set(xlabel="Retained components", ylabel="Relative integrated squared error")
    axes[1].legend()
    return fig


def plot_cluster_sweep(results, method="K-means"):
    """Compare partitions using a shared field metric across cluster counts."""
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    rows = results[results.method == method]
    for name, group in rows.groupby("representation", sort=False):
        for ax, metric in zip(axes, ["field_silhouette", "relative_field_sse"]):
            ax.plot(group.k, group[metric], marker="o", label=name)
    axes[0].set(xlabel="Clusters k", ylabel="Full-field silhouette (higher is better)")
    axes[1].set(xlabel="Clusters k", ylabel="Relative within-cluster field SSE")
    axes[1].legend(fontsize=7)
    fig.suptitle(method)
    return fig


def plot_partition_maps(scores, geometry, partitions):
    """Show partitions in identical score axes and camber coordinates.

    Horizontal offsets encode thickness within each camber coordinate so that
    all ten thicknesses remain visible. These offsets are only for display.
    Cluster colors are local to each partition and are not matched labels.
    """
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(2, len(partitions), figsize=(10, 5.5), layout="constrained")
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:len(np.unique(next(iter(partitions.values()))))])
    for column, (name, labels) in enumerate(partitions.items()):
        axes[0, column].scatter(scores[:, 0], scores[:, 1], c=labels, cmap=cmap, s=16)
        axes[0, column].set(title=name, xlabel="Functional PC1", ylabel="Functional PC2")
        offsets = 6 * (geometry[:, 2] - geometry[:, 2].mean())
        axes[1, column].scatter(100 * geometry[:, 0] + offsets, 100 * geometry[:, 1],
                               c=labels, cmap=cmap, s=12)
        axes[1, column].set(xlabel="Camber (% chord; thickness offsets)",
                            ylabel="Camber position (% chord)", xticks=[0, 2, 4, 6, 8])
    return fig


def plot_cluster_fields(fields, labels, alphas, reynolds, weights):
    """Plot complete cluster-mean surfaces and Reynolds-averaged member spread."""
    from matplotlib import pyplot as plt

    groups = np.unique(labels)
    means = np.stack([fields[labels == group].mean(axis=0) for group in groups])
    fig, axes = plt.subplots(2, len(groups), figsize=(10, 4.8), layout="constrained", squeeze=False)
    for output, name in enumerate([r"$C_D$", r"$C_L$"]):
        lo, hi = means[..., output].min(), means[..., output].max()
        for column, group in enumerate(groups):
            ax = axes[output, column]
            mesh = ax.pcolormesh(alphas, reynolds / 1e6, means[column, ..., output].T,
                                 shading="auto", cmap="viridis", vmin=lo, vmax=hi)
            ax.set(xlabel=r"$\alpha$ (deg)", ylabel=r"$Re$ ($10^6$)",
                   title=f"Cluster {group + 1}, n={np.sum(labels == group)}")
        fig.colorbar(mesh, ax=axes[output, :], label=name, shrink=0.85)
    spread, spread_axes = plt.subplots(1, 2, figsize=(7.2, 3), layout="constrained")
    curves = np.sum(fields * weights.sum(axis=0)[None, None, :, None], axis=2)
    for group in groups:
        for output, ax in enumerate(spread_axes):
            members = curves[labels == group, :, output]
            line, = ax.plot(alphas, members.mean(axis=0), label=f"Cluster {group + 1}")
            ax.fill_between(alphas, *np.quantile(members, [0.1, 0.9], axis=0),
                            color=line.get_color(), alpha=0.15)
            ax.set(xlabel=r"$\alpha$ (deg)", ylabel=[r"Re-averaged $C_D$", r"Re-averaged $C_L$"][output])
    spread_axes[1].legend()
    return fig, spread
