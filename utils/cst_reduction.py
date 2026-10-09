"""CST sampling, QoI-aware functional diffusion maps, and feasible latent decoding."""

import numpy as np
from scipy.linalg import eigh, solve
from scipy.optimize import LinearConstraint, minimize
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist, pdist
from scipy.special import comb
from scipy.stats import qmc

from utils.functional_clustering import trapezoid_weights


TE_THICKNESS = .0025


def cst_surfaces(parameters, x):
    """Upper/lower ordinates for standard degree-seven CST, N1=1/2, N2=1.

    Parameters are (sample, 16): eight upper followed by eight lower weights.
    The leading-edge modification is zero and trailing-edge thickness is fixed.
    """
    p = np.atleast_2d(parameters)
    x = np.asarray(x, dtype=float)
    if p.shape[1] != 16 or np.any((x < 0) | (x > 1)):
        raise ValueError("Sixteen coefficients and chord stations in [0,1] required.")
    basis = np.stack([comb(7, i)*x**i*(1-x)**(7-i) for i in range(8)])
    class_function = np.sqrt(x)*(1-x)
    upper = (p[:, :8] @ basis)*class_function + x*TE_THICKNESS/2
    lower = (p[:, 8:] @ basis)*class_function - x*TE_THICKNESS/2
    return upper, lower


def sample_cst(n=768, seed=42):
    """Sobol sampling of 15 independent thickness/camber shape weights.

    Half-thickness coefficients h are in [.10,.30], camber coefficients b in
    [-.06,.10]; b0=0 enforces equal upper/lower leading-edge radii. U=h+b,
    L=-h+b. A fixed sampled-geometry screen is applied before any solver call.
    """
    if not 2 <= n <= 4096:
        raise ValueError("Sample count must lie between 2 and 4096.")
    exponent = int(np.ceil(np.log2(2*n)))
    unit = qmc.Sobol(d=15, scramble=True, seed=seed).random_base2(exponent)
    h = .10 + .20*unit[:, :8]
    b = np.c_[np.zeros(len(unit)), -.06 + .16*unit[:, 8:]]
    parameters = np.c_[h+b, -h+b]
    x = (1-np.cos(np.linspace(0, np.pi, 301)))/2
    upper, lower = cst_surfaces(parameters, x)
    thickness = (upper-lower).max(axis=1)
    camber = np.abs((upper+lower)/2).max(axis=1)
    valid = (thickness >= .08) & (thickness <= .20) & (camber <= .05)
    accepted = np.flatnonzero(valid)[:n]
    if len(accepted) != n:
        raise ValueError("Sobol pool does not provide enough geometrically admissible samples.")
    return parameters[accepted], accepted, dict(proposed=len(unit),
        admissible=int(valid.sum()), rejected=int((~valid).sum()), stations=301,
        min_max_thickness=.08, max_max_thickness=.20, max_abs_camber=.05)


def evaluate_cst(parameters, alphas, reynolds, model_size="large"):
    """Evaluate NeuralFoil CD/CL/CM and retain pointwise analysis confidence."""
    import neuralfoil as nf
    a, re = np.meshgrid(alphas, reynolds, indexing="ij")
    fields = np.empty((len(parameters), len(alphas), len(reynolds), 3))
    confidence = np.empty(fields.shape[:-1])
    for i, p in enumerate(parameters):
        aero = nf.get_aero_from_kulfan_parameters(dict(upper_weights=p[:8],
            lower_weights=p[8:], leading_edge_weight=0., TE_thickness=TE_THICKNESS),
            alpha=a.ravel(), Re=re.ravel(), n_crit=9., xtr_upper=1.,
            xtr_lower=1., model_size=model_size)
        fields[i] = np.stack([aero[k] for k in ("CD", "CL", "CM")], axis=-1).reshape(fields.shape[1:])
        confidence[i] = np.asarray(aero["analysis_confidence"]).reshape(confidence.shape[1:])
    if not np.isfinite(fields).all() or np.any(fields[..., 0] <= 0):
        raise ValueError("Nonfinite or nonpositive-drag solver output; no imputation is applied.")
    if not np.isfinite(confidence).all():
        raise ValueError("Nonfinite solver confidence.")
    return fields, confidence


def operating_weights(alphas, reynolds):
    """Uniform measure in angle and log(Re), matching the sampling convention."""
    return np.outer(trapezoid_weights(alphas), trapezoid_weights(np.log(reynolds)))


def quantity_of_interest(fields, weights, name="min_drag"):
    """A minimization objective; choose grid minimum drag, mean drag or -efficiency."""
    fields = np.asarray(fields)
    if fields.shape[-1] != 3 or not np.isfinite(fields).all() or np.any(fields[..., 0] <= 0):
        raise ValueError("Finite positive-drag CD/CL/CM fields required.")
    if name == "min_drag":
        return fields[..., 0].min(axis=(1, 2))
    if name == "mean_drag":
        values = fields[..., 0]
    elif name == "negative_efficiency":
        values = np.abs(fields[..., 1])/fields[..., 0]
        return -np.einsum("nar,ar->n", values, weights)
    else:
        raise ValueError(f"Unknown quantity of interest: {name}")
    return np.einsum("nar,ar->n", values, weights)


class FieldMetric:
    """Training-only coefficient balancing under a fixed operating measure."""

    def __init__(self, weights, metric="balanced"):
        if metric not in {"balanced", "drag"}:
            raise ValueError("Metric must be balanced or drag.")
        self.weights = weights
        self.channels = np.arange(3) if metric == "balanced" else np.array([0])

    def fit(self, fields):
        self.mean = fields.mean(axis=0)
        self.scales = np.sqrt(np.einsum("arc,ar->c", fields.var(axis=0, ddof=1), self.weights))
        if np.any(self.scales <= 0):
            raise ValueError("Positive integrated coefficient variation is required.")
        return self

    def transform(self, fields):
        balanced = (fields-self.mean)*np.sqrt(self.weights)[..., None]/self.scales
        return balanced[..., self.channels].reshape(len(fields), -1)

    def relative_error(self, fields, reconstructed):
        return float(np.sum((self.transform(fields)-self.transform(reconstructed))**2)
                     / np.sum(self.transform(fields)**2))


class FunctionalDiffusionMap:
    """Dense functional kernel with optional scalar-QoI distance augmentation.

    Density normalization alpha=1 and diffusion time t=1 are fixed. Nyström
    extension uses the same full field and QoI information as the training map.
    """

    def __init__(self, dimensions=2, q_weight=0., bandwidth=.25):
        if dimensions < 1 or q_weight < 0 or bandwidth <= 0:
            raise ValueError("Invalid diffusion-map dimensions or kernel settings.")
        self.dimensions, self.q_weight, self.bandwidth = dimensions, q_weight, bandwidth

    def _distance(self, features, q):
        distance = cdist(features, self.features, metric="sqeuclidean")/self.field_distance_scale
        return distance + self.q_weight*((q[:, None]-self.q[None, :])/self.q_scale)**2

    def fit(self, features, q):
        self.features, self.q = np.asarray(features), np.asarray(q)
        self.field_distance_scale = float(np.median(pdist(features, metric="sqeuclidean")))
        self.q_scale = float(np.std(q, ddof=1))
        if self.field_distance_scale <= 0 or self.q_scale <= 0:
            raise ValueError("Nonconstant fields and QoI required.")
        distance = self._distance(self.features, self.q)
        self.epsilon = self.bandwidth*np.median(distance[np.triu_indices(len(q), 1)])
        kernel = np.exp(-distance/self.epsilon)
        self.density = kernel.sum(axis=1)
        corrected = kernel/self.density[:, None]/self.density[None, :]
        self.degree = corrected.sum(axis=1)
        symmetric = corrected/np.sqrt(self.degree[:, None]*self.degree[None, :])
        values, vectors = eigh(symmetric, subset_by_index=(len(q)-self.dimensions-1, len(q)-1))
        self.eigenvalues = values[::-1][1:]
        self.eigenfunctions = vectors[:, ::-1][:, 1:]/np.sqrt(self.degree[:, None])
        if np.min(self.eigenvalues) <= 1e-8:
            raise ValueError("Selected diffusion eigenvalues too small for extension.")
        self.coordinates = self.eigenfunctions*self.eigenvalues
        return self

    def transform(self, features, q):
        # Query density cancels in the row-normalized alpha=1 transition.
        kernel = np.exp(-self._distance(np.asarray(features), np.asarray(q))/self.epsilon)
        corrected = kernel/self.density[None, :]
        mass = corrected.sum(axis=1)
        if np.any(mass <= np.finfo(float).tiny):
            raise ValueError("Query has no kernel support; refusing extrapolation.")
        transition = corrected/mass[:, None]
        return transition @ self.eigenfunctions


def neighbor_predict(train_coordinates, query_coordinates, values, neighbors=10):
    """Inverse-distance predictions and local information retention diagnostic."""
    distance = cdist(query_coordinates, train_coordinates)
    index = np.argsort(distance, axis=1)[:, :neighbors]
    nearest = np.take_along_axis(distance, index, axis=1)
    weights = 1/np.maximum(nearest, 1e-12)**2
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ij...->i...", weights, np.asarray(values)[index]), index


class SmoothQoISurrogate:
    """RBF kernel ridge surrogate, standardized using training coordinates only."""

    def __init__(self, ridge=.01, length_scale=1.):
        self.ridge, self.length_scale = ridge, length_scale

    def fit(self, coordinates, q):
        self.center, self.scale = coordinates.mean(axis=0), coordinates.std(axis=0, ddof=1)
        if np.any(self.scale <= 0):
            raise ValueError("Latent coordinates must vary.")
        self.x = (coordinates-self.center)/self.scale
        self.q_mean, self.q_scale = q.mean(), q.std(ddof=1)
        self.gamma = 1/(self.length_scale**2*np.median(pdist(self.x, metric="sqeuclidean")))
        matrix = np.exp(-self.gamma*cdist(self.x, self.x, metric="sqeuclidean"))
        self.coefficients = solve(matrix + self.ridge*np.eye(len(q)),
                                  (q-self.q_mean)/self.q_scale, assume_a="pos")
        return self

    def predict(self, coordinates):
        x = (np.atleast_2d(coordinates)-self.center)/self.scale
        kernel = np.exp(-self.gamma*cdist(x, self.x, metric="sqeuclidean"))
        return self.q_mean+self.q_scale*(kernel @ self.coefficients)

    def value_gradient(self, z):
        delta = (z-self.center)/self.scale-self.x
        weighted = np.exp(-self.gamma*np.sum(delta**2, axis=1))*self.coefficients
        value = self.q_mean+self.q_scale*weighted.sum()
        gradient = self.q_scale*(-2*self.gamma*(weighted[:, None]*delta).sum(axis=0)/self.scale)
        return float(value), gradient


class CSTLatentDecoder:
    """Barycentric interpolation of CST coefficients inside a latent hull.

    Convex combinations preserve coefficient bounds, positive thickness, and
    equal nose radii. A separate dense-grid geometry check is still required.
    """

    def __init__(self, coordinates, parameters):
        self.coordinates, self.parameters = coordinates, parameters
        self.center, self.scale = coordinates.mean(axis=0), coordinates.std(axis=0,ddof=1)
        self.standard_coordinates = (coordinates-self.center)/self.scale
        self.triangulation = Delaunay(self.standard_coordinates)
        self.hull = ConvexHull(self.standard_coordinates)

    def decode(self, z):
        normalized = (z-self.center)/self.scale
        simplex = int(self.triangulation.find_simplex(normalized, tol=1e-9))
        if simplex < 0:
            raise ValueError("Latent point lies outside the training hull.")
        transform = self.triangulation.transform[simplex]
        bary = transform[:-1] @ (normalized-transform[-1])
        bary = np.r_[bary, 1-bary.sum()]
        if bary.min() < -1e-7:
            raise ValueError("Negative barycentric weights beyond tolerance.")
        bary = np.maximum(bary, 0)
        bary /= bary.sum()
        vertices = self.triangulation.simplices[simplex]
        return bary @ self.parameters[vertices], vertices, bary

    def optimize(self, surrogate, q, starts=8, seed=42):
        """Multistart SLSQP of a smooth scalar surrogate under latent-hull constraints."""
        rng = np.random.default_rng(seed)
        initial = [self.standard_coordinates[np.argmin(q)]]
        for _ in range(starts-1):
            vertices = rng.choice(len(q), self.coordinates.shape[1]+1, replace=False)
            initial.append(rng.dirichlet(np.ones(len(vertices))) @ self.standard_coordinates[vertices])
        constraint = LinearConstraint(self.hull.equations[:, :-1], -np.inf,
                                      -self.hull.equations[:, -1])
        results = []
        def objective(u):
            value,gradient=surrogate.value_gradient(self.center+self.scale*u)
            return value,gradient*self.scale
        for z in initial:
            fitted = minimize(objective, z, jac=True, method="SLSQP",
                              constraints=[constraint], options=dict(maxiter=200, ftol=1e-12))
            # Save failed solves too; never treat optimizer status as evidence of feasibility.
            # A declared tiny interior buffer avoids inconsistent boundary roundoff
            # between the polytope constraints and barycentric triangulation.
            u = fitted.x*(1-1e-6)
            inside = bool(self.triangulation.find_simplex(u,tol=1e-9) >= 0)
            point=self.center+self.scale*u
            results.append(dict(coordinates=point, predicted_q=float(surrogate.predict(point)[0]),
                                success=bool(fitted.success), feasible=inside,
                                iterations=int(fitted.nit), message=str(fitted.message)))
        return results
