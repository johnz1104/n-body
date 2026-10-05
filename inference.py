"""
Parameter fitting for differentiable N-body models.

JAX supplies the derivatives; SciPy handles bounded least squares.
"""

from dataclasses import dataclass
import numpy as np
from core import _jax_modules


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters and optimizer diagnostics.

    Residuals are divided by measurement uncertainty. Jacobian rows follow the
    flattened observations; columns follow parameter order. Cost history includes
    rejected trial steps. Success means the optimizer stopped successfully, not
    that the physical solution is unique.
    """

    parameters: dict
    prediction: np.ndarray
    residuals: np.ndarray
    jacobian: np.ndarray
    initial_cost: float
    cost: float
    cost_history: np.ndarray
    success: bool
    message: str
    nfev: int
    optimality: float

    @property
    def chi_squared(self) -> float:
        return 2 * self.cost


def fit_parameters(predict, initial_parameters, observations, uncertainties, *,
                   bounds=None, max_nfev=200):
    """Fit selected scalar parameters using a JAX-compatible predict(dict).

    uncertainties : Positive scalar or array broadcastable to observations.
    bounds :        Optional {name: (lower, upper)} parameter limits.

    Minimizes chi-squared / 2 for independent Gaussian errors. Keep masses
    positive with bounds or a log-mass parameter in the prediction function.
    """
    from scipy.optimize import least_squares

    jax, jnp = _jax_modules()

    # parameter values and observation errors
    names = tuple(initial_parameters)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("initial_parameters must contain named scalar parameters")
    x0 = np.asarray([initial_parameters[name] for name in names], dtype=float)
    if x0.shape != (len(names),) or not np.isfinite(x0).all():
        raise ValueError("initial parameters must be finite scalars")
    data = np.asarray(observations, dtype=float)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("observations must be nonempty and finite")
    try:
        sigma = np.broadcast_to(np.asarray(uncertainties, dtype=float), data.shape)
    except ValueError as exc:
        raise ValueError("uncertainties must broadcast to observation shape") from exc
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("uncertainties must be finite and positive")
    if (isinstance(max_nfev, (bool, np.bool_))
            or not isinstance(max_nfev, (int, np.integer))
            or max_nfev <= 0):
        raise ValueError("max_nfev must be a positive integer")

    # unspecified parameters have no bounds
    if bounds is None:
        bounds = {}
    if set(bounds) - set(names):
        raise ValueError("bounds contain unknown parameter names")
    limits = np.asarray([bounds.get(name, (-np.inf, np.inf)) for name in names], dtype=float)
    if limits.shape != (len(names), 2) or np.isnan(limits).any():
        raise ValueError("each bound must be a (lower, upper) pair without NaNs")
    lower, upper = limits.T
    if np.any(lower >= upper) or np.any(x0 < lower) or np.any(x0 > upper):
        raise ValueError("bounds must satisfy lower < upper and contain initial parameters")
    target = jnp.asarray(data)
    uncertainty = jnp.asarray(sigma)

    def residual_with_aux(x):
        parameters = dict(zip(names, x))
        prediction = jnp.asarray(predict(parameters))
        if prediction.shape != data.shape:
            raise ValueError("prediction shape must match observation shape exactly")
        residual = ((prediction - target) / uncertainty).ravel()
        return residual, residual

    # Return the residuals along with their Jacobian to avoid a second simulation.
    residual_and_jacobian = jax.jit(jax.jacrev(residual_with_aux, has_aux=True))
    previous_x = None
    previous_result = None
    history = []

    def evaluate(x):
        nonlocal previous_x, previous_result
        # SciPy requests residuals and derivatives separately at the same point.
        if previous_x is None or not np.array_equal(x, previous_x):
            jacobian, residual = residual_and_jacobian(jnp.asarray(x))
            residual = np.asarray(residual)
            jacobian = np.asarray(jacobian)
            if not np.isfinite(residual).all() or not np.isfinite(jacobian).all():
                raise ValueError("Model produced non-finite predictions or derivatives; "
                                 "check parameter bounds, timestep, and close encounters")
            previous_x = x.copy()
            previous_result = residual, jacobian
            history.append(float(0.5 * residual @ residual))
        return previous_result

    evaluate(x0)
    initial_cost = history[0]
    result = least_squares(lambda x: evaluate(x)[0], x0,
                            jac=lambda x: evaluate(x)[1], bounds=(lower, upper),
                            x_scale="jac", max_nfev=int(max_nfev),
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
    residuals = result.fun.reshape(data.shape)
    return FitResult(
        parameters=dict(zip(names, map(float, result.x))),
        prediction=data + sigma * residuals,
        residuals=residuals,
        jacobian=result.jac,
        initial_cost=initial_cost,
        cost=float(result.cost),
        cost_history=np.asarray(history),
        success=bool(result.success),
        message=str(result.message),
        nfev=int(result.nfev),
        optimality=float(result.optimality),
    )
