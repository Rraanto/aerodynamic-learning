"""Optimise airfoil shapes with AeroSandbox, CasADi, and NeuralFoil."""

import aerosandbox as asb
from aerosandbox.geometry import Airfoil, KulfanAirfoil
from aerosandbox import numpy as np


def optimise_airfoil(
    initial_airfoil: Airfoil,
    reynolds: np.ndarray,
    alpha_min: float = -20,
    alpha_max: float = 20,
) -> Airfoil:
    """
    Performs a multipoint 2D shape optimisation from an AeroSandbox airfoil.

    The angle of attack is trimmed independently at each supplied Reynolds number
    while maximising the mean lift-to-drag ratio. Aerodynamics are evaluated only
    with NeuralFoil.

    the optimisation is constrained to not allow loss in lift coefficient higher than a given threshold 
    """
    reynolds = np.array(reynolds)
    initial_kulfan = initial_airfoil.to_kulfan_airfoil(n_weights_per_side=8)

    opti = asb.Opti()
    # Broad CST bounds keep the optimizer within a numerically useful NeuralFoil range.
    upper_weights = opti.variable(
        init_guess=initial_kulfan.upper_weights,
        lower_bound=-0.5,
        upper_bound=0.5,
    )
    lower_weights = opti.variable(
        init_guess=initial_kulfan.lower_weights,
        lower_bound=-0.5,
        upper_bound=0.5,
    )
    alpha = opti.variable(
        init_guess=np.zeros(reynolds.shape),
        n_vars=np.length(reynolds),
        lower_bound=alpha_min,
        upper_bound=alpha_max,
    )

    airfoil = KulfanAirfoil(
        name=f"{initial_airfoil.name} optimized",
        upper_weights=upper_weights,
        lower_weights=lower_weights,
        leading_edge_weight=initial_kulfan.leading_edge_weight,
        TE_thickness=initial_kulfan.TE_thickness,
    )

    ## compute aero coefficients of the design airfoil
    aero = airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=reynolds)

    ## constraints and objectives 

    ## foil shape constraints 
    opti.subject_to(airfoil.local_thickness(np.linspace(0, 1, 101)) >= 0)
    opti.subject_to(airfoil.LE_radius() > 0.02)

    ## model precision constraint 
    opti.subject_to(aero["analysis_confidence"] > 0.95)

    ## aero constraints 
    #TODO: Use a more meaningful expression of the CL (eg. from a lift constraint in newtons)
    opti.subject_to(aero["CL"] > 0.5) 

    opti.minimize(np.mean(aero["CD"]))

    solution = opti.solve()
    return solution(airfoil).to_airfoil()
