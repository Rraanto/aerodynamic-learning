"""Generate NACA four-digit airfoils and their aerodynamic coefficient fields."""

from collections.abc import Iterable

import numpy 
numpy.ulong = numpy.ulonglong 
numpy.long = numpy.longlong
import aerosandbox as asb
from aerosandbox.geometry import Airfoil
import numpy as np
import pandas as pd


DEFAULT_THICKNESSES = (
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.22,
    0.24,
)
DEFAULT_CAMBERS = (0.0, 0.02, 0.04, 0.06, 0.08)
DEFAULT_CAMBER_POSITIONS = (0.2, 0.4, 0.6, 0.8)


def _naca_digits(
    values: Iterable[float], *, scale: int, minimum: int, maximum: int, name: str
) -> np.ndarray:
    """Convert chord fractions to integer NACA digits."""
    values = np.asarray(tuple(values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")

    digits = np.rint(values * scale).astype(int)
    if not np.allclose(values * scale, digits) or np.any(
        (digits < minimum) | (digits > maximum)
    ):
        raise ValueError(
            f"{name} must be representable by NACA four-digit values between "
            f"{minimum / scale:g} and {maximum / scale:g}."
        )
    return digits


def generate_airfoil_database(
    thicknesses: Iterable[float] = DEFAULT_THICKNESSES,
    cambers: Iterable[float] = DEFAULT_CAMBERS,
    camber_positions: Iterable[float] = DEFAULT_CAMBER_POSITIONS,
) -> pd.DataFrame:
    """Generate a DataFrame of unique NACA four-digit airfoil parameters.

    Parameters are expressed as chord fractions: for example, ``0.12`` thickness,
    ``0.02`` maximum camber, and ``0.4`` camber position produce NACA 2412.
    Camber position has no meaning for a symmetric airfoil, so each requested
    thickness produces only one NACA 00xx airfoil when zero camber is included.
    Each ``airfoil`` value is ``(camber, camber_position, thickness)``.
    """
    thickness_digits = _naca_digits(
        thicknesses, scale=100, minimum=1, maximum=99, name="thicknesses"
    )
    camber_digits = _naca_digits(
        cambers, scale=100, minimum=0, maximum=9, name="cambers"
    )
    position_digits = _naca_digits(
        camber_positions,
        scale=10,
        minimum=1,
        maximum=9,
        name="camber_positions",
    )

    airfoils = []
    for camber in camber_digits:
        positions = (0,) if camber == 0 else position_digits
        for position in positions:
            for thickness in thickness_digits:
                airfoils.append(
                    (
                        float(camber) / 100,
                        float(position) / 10,
                        float(thickness) / 100,
                    )
                )

    # Repeated input values should not create duplicate database entries.
    airfoils = list(dict.fromkeys(airfoils))
    return pd.DataFrame({"airfoil": airfoils})


def _make_airfoil(parameters: tuple[float, float, float]) -> Airfoil:
    camber, camber_position, thickness = parameters
    name = f"NACA{camber * 100:.0f}{camber_position * 10:.0f}{thickness * 100:02.0f}"
    return Airfoil(name)


def generate_airfoil_outputs(
    airfoils: pd.DataFrame,
    alphas: np.ndarray,
    reynolds: np.ndarray,
    method: str = "neuralfoil",
    model_size: str = "xxxlarge",
    mach: float = 0.0,
) -> pd.DataFrame:
    """Evaluate drag and lift coefficients over an operating-condition grid.

    Returns a long DataFrame with columns ``airfoil``, ``alpha``, ``Re``, ``CD``,
    and ``CL``. NeuralFoil is used by default; pass ``method="xfoil"`` to run the
    installed XFoil executable instead. XFoil points that do not converge are
    represented by ``NaN``.
    """
    if list(airfoils.columns) != ["airfoil"]:
        raise ValueError("airfoils must be a DataFrame with one 'airfoil' column.")

    airfoil_parameters = airfoils["airfoil"].tolist()
    alphas = np.asarray(alphas, dtype=float)
    reynolds = np.asarray(reynolds, dtype=float)

    if alphas.ndim != 1 or reynolds.ndim != 1:
        raise ValueError("alphas and reynolds must be one-dimensional arrays.")
    if not np.all(np.isfinite(alphas)):
        raise ValueError("alphas must contain only finite values.")
    if not np.all(np.isfinite(reynolds)) or np.any(reynolds <= 0):
        raise ValueError("reynolds must contain only finite, positive values.")
    if method not in {"neuralfoil", "xfoil"}:
        raise ValueError("method must be either 'neuralfoil' or 'xfoil'.")

    points_per_airfoil = alphas.size * reynolds.size
    outputs = pd.DataFrame(
        {
            "airfoil": [
                airfoil
                for airfoil in airfoil_parameters
                for _ in range(points_per_airfoil)
            ],
            "alpha": np.tile(
                np.repeat(alphas, reynolds.size), len(airfoil_parameters)
            ),
            "Re": np.tile(reynolds, len(airfoil_parameters) * alphas.size),
            "CD": np.nan,
            "CL": np.nan,
        }
    )
    if not airfoil_parameters or alphas.size == 0 or reynolds.size == 0:
        return outputs

    if method == "neuralfoil":
        alpha_grid, reynolds_grid = np.meshgrid(alphas, reynolds, indexing="ij")
        for airfoil_index, parameters in enumerate(airfoil_parameters):
            airfoil = _make_airfoil(parameters)
            aero = airfoil.get_aero_from_neuralfoil(
                alpha=alpha_grid.reshape(-1),
                Re=reynolds_grid.reshape(-1),
                mach=mach,
                model_size=model_size,
            )
            start = airfoil_index * points_per_airfoil
            stop = start + points_per_airfoil
            outputs.loc[start : stop - 1, "CD"] = np.asarray(aero["CD"]).reshape(-1)
            outputs.loc[start : stop - 1, "CL"] = np.asarray(aero["CL"]).reshape(-1)
        return outputs

    # Importing this module applies the local parser compatibility patch once.
    from utils import run_xfoil_patch  # noqa: F401

    for airfoil_index, parameters in enumerate(airfoil_parameters):
        airfoil = _make_airfoil(parameters)
        start = airfoil_index * points_per_airfoil
        stop = start + points_per_airfoil
        for reynolds_value in reynolds:
            aero = asb.XFoil(
                airfoil=airfoil, Re=float(reynolds_value), mach=mach
            ).alpha(alphas)
            converged_alphas = np.asarray(aero["alpha"], dtype=float)
            for result_index, converged_alpha in enumerate(converged_alphas):
                matching_rows = (
                    (outputs.index >= start)
                    & (outputs.index < stop)
                    & np.isclose(
                        outputs["alpha"], converged_alpha, rtol=0.0, atol=5e-5
                    )
                    & (outputs["Re"] == reynolds_value)
                )
                outputs.loc[matching_rows, "CD"] = aero["CD"][result_index]
                outputs.loc[matching_rows, "CL"] = aero["CL"][result_index]

    return outputs


if __name__ == "__main__":
    database = generate_airfoil_database()
    airfoil_aero = generate_airfoil_outputs(
        airfoils=database,
        alphas=np.linspace(-15, 15, 20),
        reynolds=np.linspace(1e6, 2e6, 20),
        method="neuralfoil",
        model_size="xxsmall",
    )

    airfoil_aero.to_csv("data/default_airfoils_aero.csv", index=False)
