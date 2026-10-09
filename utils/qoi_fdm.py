"""Objective definitions for the two-dimensional quantity-aware FDM study."""
import numpy as np
from utils.cst_reduction import operating_weights

NAMES = ('min_drag', 'best_drag_lift', 'mean_drag', 'multipoint_drag',
         'worst_drag', 'drag_variability', 'mean_efficiency', 'best_endurance',
         'moment_rms', 'lift_tracking', 'trimmed_drag', 'trimmed_worst_drag')


def objectives(fields, alpha, reynolds):
    """Return minimization objectives; prescribed-lift drag uses first rising crossing.

    Multipoint conditions: alpha=0,4,8 degrees at Re=.3,1,3 million.
    Trimmed conditions: CL=.4,.7,1 at the same Reynolds numbers.
    No extrapolation or sorting of nonmonotone lift curves is allowed.
    """
    cd, cl, cm = np.moveaxis(np.asarray(fields), -1, 0)
    w = operating_weights(alpha, reynolds)
    mean = lambda v: np.einsum('nar,ar->n', v, w)
    def at_re(v, re):
        return np.stack([np.interp(np.log(re), np.log(reynolds), row) for row in v.reshape(-1, len(reynolds))]).reshape(v.shape[:2])
    multipoint, trimmed = [], []
    for re in (3e5, 1e6, 3e6):
        d, l = at_re(cd, re), at_re(cl, re)
        for a in (0., 4., 8.):
            multipoint.append(np.array([np.interp(a, alpha, row) for row in d]))
        for target in (.4, .7, 1.):
            values = np.full(len(cd), np.nan)
            for i in range(len(cd)):
                crossings = np.flatnonzero((l[i, :-1] <= target) & (l[i, 1:] >= target) & (np.diff(l[i]) > 0))
                if len(crossings):
                    j = crossings[0]
                    t = (target-l[i,j])/(l[i,j+1]-l[i,j])
                    values[i] = (1-t)*d[i,j]+t*d[i,j+1]
            trimmed.append(values)
    result = [cd.min((1,2)), np.min(np.where(cl > 0, cd/np.maximum(cl,1e-12), np.inf),(1,2)),
              mean(cd), np.mean(multipoint,axis=0), cd.max((1,2)),
              np.sqrt(mean((cd-mean(cd)[:,None,None])**2)),
              -mean(np.maximum(cl,0)/cd), -np.max(np.maximum(cl,0)**1.5/cd,axis=(1,2)),
              np.sqrt(mean(cm**2)), np.sqrt(mean((cl-.7)**2)),
              np.mean(trimmed,axis=0), np.max(trimmed,axis=0)]
    return dict(zip(NAMES,result))
