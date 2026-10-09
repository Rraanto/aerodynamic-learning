"""Contour-neighborhood and operating-regime diagnostics for cached airfoil fields."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

from utils.functional_clustering import trapezoid_weights


def interval_weights(grid, lower, upper):
    """Integrate piecewise-linear nodal values over a clipped interval.

    Weights are unnormalized; splitting intervals preserves the full integral.
    """
    grid = np.asarray(grid, dtype=float)
    trapezoid_weights(grid)
    if not grid[0] <= lower < upper <= grid[-1]:
        raise ValueError("Interval must lie inside the grid and have positive width.")
    weights = np.zeros(len(grid))
    for i, (left, right) in enumerate(zip(grid[:-1], grid[1:])):
        a, b = max(left, lower), min(right, upper)
        if b <= a:
            continue
        right_weight = ((b - left)**2 - (a - left)**2) / (2 * (right - left))
        weights[i] += b - a - right_weight
        weights[i + 1] += right_weight
    return weights


def operating_regimes(alphas, reynolds, angle_edges=(0., 5., 10., 15.), sign="both"):
    """Nine normalized regime weights, uniform in angle and linear Reynolds.

    Magnitude bins combine both incidence signs unless explicitly restricted.
    """
    alphas, reynolds = np.asarray(alphas), np.asarray(reynolds)
    edges = np.asarray(angle_edges, dtype=float)
    if len(edges) != 4 or edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError("Four increasing angle edges starting at zero are required.")
    if sign not in {"both", "positive", "negative"}:
        raise ValueError("Unknown incidence sign.")
    re_edges = np.linspace(reynolds[0], reynolds[-1], 4)
    names = ("low", "middle", "high")
    regimes = []
    for a, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        wa = np.zeros(len(alphas))
        if sign in {"both", "positive"}:
            wa += interval_weights(alphas, lo, hi)
        if sign in {"both", "negative"}:
            wa += interval_weights(alphas, -hi, -lo)
        for r, (rlo, rhi) in enumerate(zip(re_edges[:-1], re_edges[1:])):
            wr = interval_weights(reynolds, rlo, rhi)
            volume = wa.sum() * wr.sum()
            regimes.append(dict(name=f"{names[a]}-{names[r]}", angle=names[a],
                                reynolds=names[r], alpha_lower=lo, alpha_upper=hi,
                                re_lower=rlo, re_upper=rhi, volume=volume,
                                weights=np.outer(wa, wr) / volume))
    return regimes


def regime_performance(fields, regimes):
    """Per-airfoil mean drag, signed lift, lift magnitude, and mean |CL|/CD.

    Efficiency is the mean of pointwise ratios, not a ratio of mean coefficients.
    No positive-lift mission, trim constraint, or stall threshold is assumed.
    """
    fields = np.asarray(fields, dtype=float)
    if fields.ndim != 4 or fields.shape[-1] != 2 or not np.isfinite(fields).all():
        raise ValueError("Finite fields with shape (airfoil, alpha, Re, 2) required.")
    cd, cl = fields[..., 0], fields[..., 1]
    if np.any(cd <= 0):
        raise ValueError("Positive drag is required for efficiency.")
    quantities = np.stack([cd, cl, np.abs(cl), np.abs(cl) / cd], axis=-1)
    return np.stack([np.einsum("narc,ar->nc", quantities, r["weights"])
                     for r in regimes], axis=1)


def naca_contours(geometry, stations=201):
    """Registered upper/lower NACA four-digit coordinates in chord units.

    Normal thickness offsets and the original finite trailing-edge convention
    match the family generator used by AeroSandbox.
    """
    if stations < 3:
        raise ValueError("At least three contour stations required.")
    x = (1 - np.cos(np.linspace(0, np.pi, stations))) / 2
    contours = []
    for m, p, t in np.asarray(geometry):
        if m == 0:
            yc, slope = np.zeros_like(x), np.zeros_like(x)
        else:
            if not 0 < p < 1:
                raise ValueError("Cambered airfoils require 0 < p < 1.")
            front = x < p
            yc = np.where(front, m / p**2 * (2*p*x - x*x),
                          m / (1-p)**2 * (1 - 2*p + 2*p*x - x*x))
            slope = np.where(front, 2*m / p**2 * (p-x), 2*m / (1-p)**2 * (p-x))
        yt = 5*t*(.2969*np.sqrt(x) - .1260*x - .3516*x**2 + .2843*x**3 - .1015*x**4)
        theta = np.arctan(slope)
        contours.append(np.stack([x-yt*np.sin(theta), yc+yt*np.cos(theta),
                                  x+yt*np.sin(theta), yc-yt*np.cos(theta)], axis=-1))
    return x, np.asarray(contours)


def contour_features(geometry, stations=201):
    """Euclidean features whose distance is registered contour RMS / chord."""
    x, contours = naca_contours(geometry, stations)
    return (contours * np.sqrt(trapezoid_weights(x)[None, :, None] / 2)).reshape(len(contours), -1)


def neighborhood_diagnostics(reference, embedding, neighbors=5):
    """Descriptive distance correlation and directed nearest-neighbor overlap."""
    if not 1 <= neighbors < len(reference):
        raise ValueError("Neighbor count must be between 1 and n-1.")
    dg, de = cdist(reference, reference), cdist(embedding, embedding)
    pairs = np.triu_indices(len(reference), 1)
    correlation = float(spearmanr(dg[pairs], de[pairs]).statistic)
    np.fill_diagonal(dg, np.inf)
    np.fill_diagonal(de, np.inf)
    ng, ne = np.argsort(dg, axis=1)[:, :neighbors], np.argsort(de, axis=1)[:, :neighbors]
    overlap = np.array([len(set(a) & set(b)) / neighbors for a, b in zip(ng, ne)])
    return dict(distance_spearman=correlation, neighbor_overlap=float(overlap.mean()),
                chance_overlap=neighbors / (len(reference)-1)), overlap, ng, ne


def weighted_field_features(fields, weights, scales):
    """Regime-restricted fields using globally fixed coefficient scales."""
    return ((fields - fields.mean(axis=0)) * np.sqrt(weights)[None, ..., None]
            / scales).reshape(len(fields), -1)


def relative_cluster_sse(features, labels):
    """Centroid error divided by population-centered energy in a fixed metric."""
    centered = features - features.mean(axis=0)
    total = np.sum(centered**2)
    if total == 0:
        raise ValueError("Cluster SSE requires nonzero field variance.")
    return float(sum(np.sum((features[labels == k] - features[labels == k].mean(axis=0))**2)
                     for k in np.unique(labels)) / total)
