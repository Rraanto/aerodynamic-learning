"""Reproduce geometry and regime analysis without rerunning aerodynamic solvers.

Run from the repository root: python notebooks/airfoil_regimes.py
"""

import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.airfoil_regimes import (contour_features, naca_contours,
    neighborhood_diagnostics, operating_regimes, regime_performance,
    relative_cluster_sse, weighted_field_features)


SOURCE = ROOT / "data/multivariate_pca_clustering"
DATA = ROOT / "data/airfoil_regimes"
FIGURES = ROOT / "figures/airfoil_regimes"
SEED = 42
K = 4


def cluster(features, k=K):
    model = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=SEED)
    return model.fit_predict(features)


def save_figure(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=180)
    plt.close(fig)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use(ROOT / "utils/academic.mplstyle")
    metadata = json.loads((SOURCE / "run_metadata.json").read_text())
    cache = SOURCE / metadata["cache"]
    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    if digest != metadata["cache_sha256"]:
        raise ValueError("Field cache differs from recorded source.")
    with np.load(cache) as z:
        fields, geometry = z["fields"], z["geometry"]
        alphas, reynolds = z["alphas"], z["reynolds"]
        if z["coefficients"].tolist() != ["CD", "CL"]:
            raise ValueError("Expected CD, CL channel order.")
    with np.load(SOURCE / "functional_pca.npz") as z:
        scores, scales = z["scores"], z["scales"]
        mean, components, weights = z["mean"], z["components"], z["weights"]
        eigenvalues = z["explained_variance"]
    assignments = pd.read_csv(SOURCE / "assignments.csv")
    np.testing.assert_allclose(geometry, assignments[["camber", "camber_position", "thickness"]])
    names = assignments.airfoil.to_numpy()
    full = weighted_field_features(fields, weights, scales)
    # Check that saved coordinates still represent this cache with the same signs.
    np.testing.assert_allclose(scores, full @ components[:scores.shape[1]].T, atol=1e-9)
    shape = contour_features(geometry)
    diag_rows, overlaps = [], {}
    for name, features in [("global2D", scores[:, :2]), ("global7D", scores), ("full fields", full)]:
        for nn in (1, 5, 10):
            result, overlap, _, _ = neighborhood_diagnostics(shape, features, nn)
            diag_rows.append(dict(representation=name, neighbors=nn, **result))
            if nn == 5:
                overlaps[name] = overlap
    pd.DataFrame(diag_rows).to_csv(DATA / "geometry_neighborhoods.csv", index=False)
    pc_rows = [dict(parameter=parameter, PC=pc+1,
                    spearman=float(spearmanr(geometry[:, j], scores[:, pc]).statistic))
               for j, parameter in enumerate(("camber", "position", "thickness")) for pc in range(2)]
    pd.DataFrame(pc_rows).to_csv(DATA / "pc_geometry.csv", index=False)

    regimes = operating_regimes(alphas, reynolds)
    performance = regime_performance(fields, regimes)
    # Equal regime and scalar weights after population standardization.
    profiles = performance[..., [0, 2, 3]].reshape(len(fields), -1)
    scaler = StandardScaler().fit(profiles)
    profile_features = scaler.transform(profiles)
    regime_labels = cluster(profile_features)
    # Local numbering sorted by cluster mean camber, then thickness.
    order = sorted(np.unique(regime_labels), key=lambda k: tuple(geometry[regime_labels == k].mean(axis=0)[[0, 2]]))
    regime_labels = np.array([order.index(k) for k in regime_labels])
    partitions = {
        "global7D": assignments["MFPCA/K-means/k=4"].to_numpy(),
        "global2D": assignments["MFPCA-2D/K-means/k=4"].to_numpy(),
        "geometry": assignments["Geometry (m,p,t)/K-means/k=4"].to_numpy(),
        "regime profile": regime_labels,
    }
    profile_pca = PCA(n_components=2).fit(profile_features)
    profile_scores = profile_pca.transform(profile_features)
    reconstructed2 = (scores[:, :2] @ components[:2]).reshape(fields.shape)
    reconstructed2 = reconstructed2 / (np.sqrt(weights)[..., None] / scales) + mean
    reconstructed7 = (scores @ components[:scores.shape[1]]).reshape(fields.shape)
    reconstructed7 = reconstructed7 / (np.sqrt(weights)[..., None] / scales) + mean
    rows, perf_rows, local_labels, regime_info = [], [], [], []
    for r, regime in enumerate(regimes):
        local = weighted_field_features(fields, regime["weights"], scales)
        labels = cluster(local)
        local_labels.append(labels)
        local_eigen = np.linalg.eigvalsh(local @ local.T)[::-1]
        variance2 = local_eigen[:2].sum() / local_eigen.sum()
        for representation, labels_here in {**partitions, "local fields": labels}.items():
            rows.append(dict(regime=regime["name"], representation=representation,
                             relative_sse=relative_cluster_sse(local, labels_here),
                             silhouette=silhouette_score(local, labels_here),
                             ARI_global7D=adjusted_rand_score(partitions["global7D"], labels_here)))
        residual2 = ((fields-reconstructed2)*np.sqrt(regime["weights"])[..., None]/scales).reshape(len(fields), -1)
        residual7 = ((fields-reconstructed7)*np.sqrt(regime["weights"])[..., None]/scales).reshape(len(fields), -1)
        energy = np.sum(local**2)
        regime_info.append({k: v for k, v in regime.items() if k != "weights"} | dict(
            global2D_relative_error=float(np.sum(residual2**2)/energy),
            global7D_relative_error=float(np.sum(residual7**2)/energy),
            local2D_variance=float(variance2)))
        for i in range(len(fields)):
            perf_rows.append(dict(airfoil=names[i], regime=regime["name"],
                                  cluster=int(regime_labels[i]+1),
                                  CD=performance[i, r, 0], CL=performance[i, r, 1],
                                  absCL=performance[i, r, 2], efficiency=performance[i, r, 3]))
    pd.DataFrame(regime_info).to_csv(DATA / "regimes.csv", index=False)
    pd.DataFrame(rows).to_csv(DATA / "cluster_metrics.csv", index=False)
    perf_table = pd.DataFrame(perf_rows)
    perf_table.to_csv(DATA / "performance.csv", index=False)
    summaries = perf_table.groupby(["cluster", "regime"], sort=False)[["CD", "CL", "absCL", "efficiency"]].agg(["mean", "median", lambda a: a.quantile(.1), lambda a: a.quantile(.9)])
    summaries.columns = [f"{a}_{b}".replace("<lambda_0>", "q10").replace("<lambda_1>", "q90") for a, b in summaries.columns]
    summaries.to_csv(DATA / "cluster_performance.csv")
    points = assignments[["airfoil", "camber", "camber_position", "thickness"]].copy()
    points["PC1"], points["PC2"] = scores[:, 0], scores[:, 1]
    for name, labels in partitions.items():
        points[name] = labels + 1
    for name, overlap in overlaps.items():
        points[f"shape_overlap5_{name}"] = overlap
    for r, labels in enumerate(local_labels):
        points[f"local_{regimes[r]['name']}"] = labels + 1
    points.to_csv(DATA / "points.csv", index=False)
    cluster_geometry = points.groupby("regime profile").agg(
        n=("airfoil", "size"), camber_mean=("camber", "mean"), camber_min=("camber", "min"),
        camber_max=("camber", "max"), thickness_mean=("thickness", "mean"),
        thickness_min=("thickness", "min"), thickness_max=("thickness", "max"))
    cluster_geometry.to_csv(DATA / "cluster_geometry.csv")

    sensitivity = []
    for sign in ("positive", "negative"):
        p = regime_performance(fields, operating_regimes(alphas, reynolds, sign=sign))
        labels = cluster(StandardScaler().fit_transform(p[..., [0, 2, 3]].reshape(len(fields), -1)))
        sensitivity.append(dict(check=sign, ARI=adjusted_rand_score(regime_labels, labels)))
    for edges in ((0, 4, 9, 15), (0, 6, 11, 15)):
        p = regime_performance(fields, operating_regimes(alphas, reynolds, edges))
        labels = cluster(StandardScaler().fit_transform(p[..., [0, 2, 3]].reshape(len(fields), -1)))
        sensitivity.append(dict(check=f"angle edges {edges}", ARI=adjusted_rand_score(regime_labels, labels)))
    for k in (3, 5):
        labels = cluster(profile_features, k)
        sensitivity.append(dict(check=f"k={k}", ARI=adjusted_rand_score(regime_labels, labels)))
    for r, regime in enumerate(regimes):
        keep = np.delete(np.arange(27), np.arange(3*r, 3*r+3))
        labels = cluster(profile_features[:, keep])
        sensitivity.append(dict(check=f"omit {regime['name']}", ARI=adjusted_rand_score(regime_labels, labels)))
    rng = np.random.default_rng(SEED)
    for repeat in range(10):
        subset = rng.choice(len(fields), int(.8*len(fields)), replace=False)
        sub_scaler = StandardScaler().fit(profiles[subset])
        model = KMeans(n_clusters=K, n_init=20, max_iter=500, random_state=SEED+repeat)
        model.fit(sub_scaler.transform(profiles[subset]))
        labels = model.predict(sub_scaler.transform(profiles))
        sensitivity.append(dict(check=f"subsample {repeat}", ARI=adjusted_rand_score(regime_labels, labels)))
    pd.DataFrame(sensitivity).to_csv(DATA / "sensitivity.csv", index=False)
    agreement = pd.DataFrame([[adjusted_rand_score(a, b) for b in partitions.values()]
                              for a in partitions.values()], index=partitions, columns=partitions)
    agreement.to_csv(DATA / "partition_agreement.csv")

    dg, de = cdist(shape, shape), cdist(scores[:, :2], scores[:, :2])
    indices = np.triu_indices(len(fields), 1)
    pairs = pd.DataFrame(dict(i=indices[0], j=indices[1],
                             shape_RMS=dg[indices], distance2D=de[indices]))
    pairs["shape_percentile"] = pairs.shape_RMS.rank(pct=True)
    pairs["map_percentile"] = pairs.distance2D.rank(pct=True)
    pairs["airfoil_i"], pairs["airfoil_j"] = names[pairs.i], names[pairs.j]
    pairs.to_csv(DATA / "pair_distances.csv", index=False)
    examples = [pairs.loc[pairs.shape_percentile <= .05].sort_values("distance2D").iloc[-1],
                pairs.loc[pairs.map_percentile <= .05].sort_values("shape_RMS").iloc[-1]]
    pd.DataFrame(examples).to_csv(DATA / "example_pairs.csv", index=False)
    make_figures(scores, geometry, regimes, performance, partitions, local_labels,
                 profile_scores, examples, names, overlaps, pd.DataFrame(rows))
    summary = dict(n_airfoils=len(fields), global2D_variance=float(eigenvalues[:2].sum()/eigenvalues.sum()),
                   profile2D_variance=float(profile_pca.explained_variance_ratio_.sum()),
                   profile_silhouette=float(silhouette_score(profile_features, regime_labels)),
                   profile_relative_sse=relative_cluster_sse(profile_features, regime_labels),
                   geometry_neighborhoods=diag_rows, pc_geometry=pc_rows,
                   agreement=agreement.to_dict(), sensitivity=sensitivity,
                   regime_diagnostics=regime_info,
                   examples=pd.DataFrame(examples).to_dict(orient="records"))
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    write_report_tables(summary, cluster_geometry, pd.DataFrame(rows))
    run = dict(seed=SEED, k=K, source_cache=str(cache.relative_to(ROOT)), cache_sha256=digest,
               python=platform.python_version(),
               versions={p: version(p) for p in ("numpy", "scipy", "pandas", "matplotlib", "scikit-learn")},
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__), ROOT/"utils/airfoil_regimes.py", SOURCE/"functional_pca.npz", SOURCE/"assignments.csv")},
               geometry_stations=201, performance_features=["mean_CD", "mean_absCL", "mean_absCL_over_CD"],
               angle_edges=[0, 5, 10, 15], reynolds_edges=np.linspace(reynolds[0], reynolds[-1], 4).tolist(),
               weighting="piecewise-linear quadrature, uniform alpha and linear Re; equal standardized regime metrics",
               source_generation=json.loads(str(np.load(cache)["config"])))
    (DATA / "run_metadata.json").write_text(json.dumps(run, indent=2)+"\n")
    print(json.dumps({k: summary[k] for k in ("n_airfoils", "global2D_variance", "profile2D_variance", "profile_silhouette", "profile_relative_sse")}, indent=2))
    print(agreement.round(3).to_string())
    print(pd.DataFrame(regime_info)[["name", "global2D_relative_error", "global7D_relative_error", "local2D_variance"]].round(3).to_string(index=False))


def write_report_tables(summary, geometry, metrics):
    report = ROOT / "report"
    report.mkdir(exist_ok=True)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Regime & $E_2$ & $E_7$ & Local $V_2$ & Global SSE & Local SSE & ARI\\",
             r"\midrule"]
    for r in summary["regime_diagnostics"]:
        table = metrics[metrics.regime == r["name"]].set_index("representation")
        lines.append(f"{r['name']} & {r['global2D_relative_error']:.3f} & {r['global7D_relative_error']:.3f} & "
                     f"{r['local2D_variance']:.3f} & {table.loc['global7D', 'relative_sse']:.3f} & "
                     f"{table.loc['local fields', 'relative_sse']:.3f} & {table.loc['local fields', 'ARI_global7D']:.3f}" + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (report / "regime_table.tex").write_text("\n".join(lines)+"\n")
    lines = [r"\begin{tabular}{lrrrl}", r"\toprule",
             r"Cluster & $n$ & Mean $m/c$ (\%) & Mean $t/c$ (\%) & Interpretation\\", r"\midrule"]
    descriptions = ["Low drag; strongest high-angle efficiency", "Intermediate camber and efficiency",
                    "Thin; high-angle drag penalty", "High camber; strongest low-angle efficiency"]
    for (k, row), description in zip(geometry.iterrows(), descriptions):
        lines.append(f"R{k} & {int(row.n)} & {100*row.camber_mean:.2f} & {100*row.thickness_mean:.2f} & {description}" + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (report / "cluster_table.tex").write_text("\n".join(lines)+"\n")


def make_figures(scores, geometry, regimes, performance, partitions, local_labels,
                 profile_scores, examples, names, overlaps, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), layout="constrained", sharex=True, sharey=True)
    for j, ax in enumerate(axes):
        for m, p in np.unique(geometry[:, :2], axis=0):
            selected = (geometry[:, 0] == m) & (geometry[:, 1] == p)
            ax.plot(scores[selected, 0], scores[selected, 1], color=".8", lw=.6, zorder=0)
        points = ax.scatter(*scores[:, :2].T, c=100*geometry[:, j], s=12, cmap="viridis")
        fig.colorbar(points, ax=ax, label=["Camber (% chord)", "Camber position (% chord)", "Thickness (% chord)"][j], orientation="horizontal", pad=.16)
        ax.set(xlabel="Functional PC1", title=["(a) Camber", "(b) Position", "(c) Thickness"][j])
    axes[0].set_ylabel("Functional PC2")
    save_figure(fig, "geometry_map")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), layout="constrained")
    x, contours = naca_contours(geometry)
    for row, pair in enumerate(examples):
        i, j = int(pair.i), int(pair.j)
        ax = axes[row, 0]
        ax.scatter(*scores[:, :2].T, s=8, color=".8")
        for idx, marker, color in ((i, "o", "tab:blue"), (j, "s", "tab:orange")):
            ax.scatter(*scores[idx, :2], marker=marker, color=color, s=45, label=names[idx])
            c = contours[idx]
            axes[row, 1].plot(c[:, 0], c[:, 1], color=color, label=names[idx])
            axes[row, 1].plot(c[:, 2], c[:, 3], color=color)
        ax.plot(scores[[i,j], 0], scores[[i,j], 1], color=".4", ls="--")
        ax.set(xlabel="PC1", ylabel="PC2", title=["Similar contours, separated responses", "Close responses, different contours"][row])
        ax.legend(loc="upper left", fontsize=7)
        axes[row, 1].set(xlabel="x/c", ylabel="y/c", title=f"Contour RMS = {pair.shape_RMS:.4f} chord")
        axes[row, 1].set_aspect("equal", adjustable="datalim")
    save_figure(fig, "contour_examples")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    for ax, coords, title in zip(axes, (scores[:, :2], profile_scores),
                                 ("Regime clusters on the global map", "PCA of 27 performance descriptors")):
        for k in range(K):
            member = partitions["regime profile"] == k
            ax.scatter(*coords[member].T, label=f"R{k+1}, n={member.sum()}", s=18)
        ax.set(xlabel="PC1", ylabel="PC2", title=title)
    axes[1].legend(fontsize=7)
    save_figure(fig, "regime_clusters")
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6), layout="constrained", sharex=True, sharey=True)
    for r, ax in enumerate(axes.flat):
        ax.scatter(*scores[:, :2].T, c=local_labels[r], cmap="tab10", vmin=0, vmax=9, s=10)
        ari = metrics.query("representation == 'local fields'").iloc[r].ARI_global7D
        ax.set(title=f"{regimes[r]['name']}; ARI={ari:.2f}")
        if r//3 == 2:
            ax.set_xlabel("Global PC1")
        if r%3 == 0:
            ax.set_ylabel("Global PC2")
    save_figure(fig, "local_partitions")
    means = np.stack([performance[partitions["regime profile"] == k].mean(axis=0) for k in range(K)])
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), layout="constrained", sharex=True)
    for ax, output, title in zip(axes, (0, 2, 3), (r"Mean $C_D$ (lower)", r"Mean $|C_L|$ (higher)", r"Mean $|C_L|/C_D$ (higher)")):
        mesh = ax.imshow(means[:, :, output], aspect="auto", cmap="viridis")
        for k in range(K):
            for r in range(9):
                value = means[k, r, output]
                normalized = mesh.norm(value)
                ax.text(r, k, f"{value:.3f}" if output == 0 else f"{value:.1f}", ha="center", va="center",
                        fontsize=7, color="white" if normalized < .5 else "black")
        ax.set(yticks=range(K), yticklabels=[f"R{k+1}" for k in range(K)], title=title)
        fig.colorbar(mesh, ax=ax, shrink=.9)
    axes[-1].set(xticks=range(9), xticklabels=[r["name"] for r in regimes], xlabel="Angle magnitude – Reynolds regime")
    axes[-1].tick_params(axis="x", rotation=30)
    save_figure(fig, "performance")
    fig, ax = plt.subplots(figsize=(7.2, 3), layout="constrained")
    for name, group in metrics.groupby("representation", sort=False):
        ax.plot(range(9), group.relative_sse, "o-", label=name)
    ax.set(xticks=range(9), xticklabels=[r["name"] for r in regimes], ylabel="Relative regime field SSE", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(ncol=3, fontsize=7)
    save_figure(fig, "regime_errors")


if __name__ == "__main__":
    main()
