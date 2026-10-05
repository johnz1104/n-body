"""Independent trajectory, derivative, observation, and inference regressions."""

import json
import subprocess
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from core import (Body, ParticleState, enable_autodiff, make_state, barycentric_state,
                  differentiable_accelerations, differentiable_energy)
from universe import Universe, Trajectory, simulate
from observables import radial_velocity, sky_plane_positions, sample_observable
from inference import fit_parameters

enable_autodiff()
import jax.numpy as jnp


def binary_state():
    return make_state([0.4, 0.6], [[-0.6, 0.1, 0.03], [0.4, 0, 0]],
                      [[0.03, -0.6, 0.02], [0, 0.4, 0.01]])


@pytest.mark.parametrize("method", ["rk4", "leapfrog"])
@pytest.mark.parametrize("epsilon", [0.0, 0.1])
def test_jax_trajectories_match_numpy_without_mutation(method, epsilon):
    state = binary_state()
    u = Universe(dt=0.01, G=1, epsilon=epsilon)
    for i in range(2):
        u.add_body(Body(str(i), state.masses[i], state.positions[i], state.velocities[i]))
    trajectory = u.simulate_differentiable(40, method)
    assert trajectory.positions.dtype == jnp.float64
    assert u.time == 0 and u.steps == 0 and u.diag_time == []
    np.testing.assert_array_equal([b.position for b in u.bodies], state.positions)
    assert all(b.history_pos == [] for b in u.bodies)
    compiled = jax.jit(lambda s: simulate(s, dt=0.01, steps=40, method=method,
                                         epsilon=epsilon))(state)
    u.run(40, method, progress=False)
    expected_r = np.stack([b.history_pos for b in u.bodies], axis=1)
    expected_v = np.stack([b.history_vel for b in u.bodies], axis=1)
    np.testing.assert_allclose(trajectory.positions, expected_r, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(trajectory.velocities, expected_v, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(compiled.positions, trajectory.positions, atol=2e-14)
    energies = jax.vmap(lambda r, v: differentiable_energy(
        ParticleState(state.masses, r, v), epsilon=epsilon))(
            trajectory.positions, trajectory.velocities)
    np.testing.assert_allclose(energies, u.diag_E, atol=1e-14)
    continuation = u.simulate_differentiable(0)
    assert continuation.times[0] == pytest.approx(u.time)
    np.testing.assert_allclose(continuation.positions[0], expected_r[-1])


@pytest.mark.parametrize("method", ["rk4", "leapfrog"])
@pytest.mark.parametrize("epsilon", [0.0, 0.08])
def test_mass_position_velocity_gradients_against_finite_differences(method, epsilon):
    state = binary_state()

    def objective(s):
        trajectory = simulate(s, dt=0.02, steps=30, method=method, epsilon=epsilon)
        rv = radial_velocity(trajectory, line_of_sight=(1, 2, 3))
        sky = sky_plane_positions(trajectory, body_index=1, reference_index=0)
        return rv[-1] + 0.3*jnp.sum(sky[-1]**2)

    gradient = jax.jit(jax.grad(objective))(state)
    rng = np.random.default_rng(14)
    for i, component in enumerate(state):
        direction = rng.normal(size=component.shape)
        direction /= np.linalg.norm(direction)
        plus, minus = list(state), list(state)
        plus[i] = component + 1e-5*direction
        minus[i] = component - 1e-5*direction
        finite_difference = (objective(ParticleState(*plus))-objective(ParticleState(*minus)))/2e-5
        derivative = jnp.sum(gradient[i]*direction)
        assert np.isfinite(gradient[i]).all()
        assert abs(float(derivative)) > 1e-5
        np.testing.assert_allclose(derivative, finite_difference, rtol=2e-7, atol=2e-9)


@pytest.mark.parametrize("method,order_ratio", [("rk4", 16), ("leapfrog", 4)])
def test_derivative_converges_to_analytic_circular_orbit(method, order_ratio):
    # A family of exact circular binaries with separation 1 and total mass M.
    def final_y(mass, steps):
        state = make_state(jnp.array([mass/2, mass/2]),
            [[-0.5, 0, 0], [0.5, 0, 0]],
            jnp.array([[0, -0.5, 0], [0, 0.5, 0]])*jnp.sqrt(mass))
        return simulate(state, dt=1/steps, steps=steps, method=method).positions[-1, 1, 1]

    exact = 0.25*np.cos(1.0)
    errors = [abs(float(jax.grad(lambda m: final_y(m, n))(1.0))-exact) for n in (20, 40)]
    assert order_ratio*0.8 < errors[0]/errors[1] < order_ratio*1.2


def test_force_is_negative_energy_gradient_and_self_derivative_is_finite():
    state = binary_state()
    force = -jax.grad(lambda r: differentiable_energy(state._replace(positions=r), epsilon=0.1))(state.positions)
    acceleration = differentiable_accelerations(state.masses, state.positions, epsilon=0.1)
    np.testing.assert_allclose(force, state.masses[:, None]*acceleration, atol=2e-15)
    single = make_state([2.0], [[1, 2, 3]], [[0.1, 0.2, 0.3]])
    trajectory = simulate(single, dt=0.1, steps=10)
    np.testing.assert_allclose(trajectory.positions[-1], [[1.1, 2.2, 3.3]], atol=1e-14)
    derivative = jax.grad(lambda s: jnp.sum(simulate(s, dt=0.1, steps=10).positions[-1]))(single)
    np.testing.assert_array_equal(derivative.masses, [0.0])
    np.testing.assert_allclose(derivative.positions, 1, atol=1e-14)
    np.testing.assert_allclose(derivative.velocities, 1, atol=1e-14)


def test_leapfrog_conserves_momentum_and_bounds_energy_error():
    state = make_state([0.5, 0.5], [[-0.5, 0, 0], [0.5, 0, 0]],
                       [[0, -0.5, 0], [0, 0.5, 0]])
    trajectory = simulate(state, dt=0.02, steps=1500)
    energies = jax.vmap(lambda r, v: differentiable_energy(state._replace(positions=r, velocities=v)))(
        trajectory.positions, trajectory.velocities)
    assert np.max(np.abs((energies-energies[0])/energies[0])) < 1e-6
    np.testing.assert_allclose(jnp.sum(state.masses[None, :, None]*trajectory.velocities, axis=1), 0, atol=1e-14)


def test_barycentric_transform_and_batched_simulations():
    state = barycentric_state(binary_state())
    np.testing.assert_allclose(jnp.sum(state.masses[:, None]*state.positions, axis=0), 0, atol=1e-16)
    np.testing.assert_allclose(jnp.sum(state.masses[:, None]*state.velocities, axis=0), 0, atol=1e-16)
    def run(velocity):
        return simulate(state._replace(velocities=velocity), dt=0.01, steps=4).positions
    batch = jax.jit(jax.vmap(run))(jnp.stack([state.velocities, 1.1*state.velocities]))
    assert batch.shape == (2, 5, 2, 3)
    np.testing.assert_allclose(batch[1], run(1.1*state.velocities), atol=1e-14)


def test_observable_geometry_and_interpolation_derivatives():
    trajectory = Trajectory(jnp.array([0., 1., 2.]),
        jnp.array([[[1., 2, 3], [3, 6, 9]], [[2, 4, 6], [4, 8, 12]], [[3, 6, 9], [5, 10, 15]]]),
        jnp.broadcast_to(jnp.array([[1., 2, 3], [4, 5, 6]]), (3, 2, 3)))
    np.testing.assert_allclose(radial_velocity(trajectory, line_of_sight=(0, 0, 2), systemic_velocity=7), 10)
    np.testing.assert_allclose(sky_plane_positions(trajectory, 1, reference_index=0, distance=2),
                               np.tile([1, 2], (3, 1)))
    values = jnp.array([[0., 10.], [2., 12.], [4., 14.]])
    np.testing.assert_allclose(sample_observable(trajectory, values, [0.5, 1.5]), [[1, 11], [3, 13]])
    gradient = jax.grad(lambda x: jnp.sum(sample_observable(trajectory, x, [0.25])))(values)
    np.testing.assert_allclose(gradient, [[0.75, 0.75], [0.25, 0.25], [0, 0]])
    with pytest.raises(ValueError, match="inside"):
        sample_observable(trajectory, values, [-0.1])
    compiled = jax.jit(lambda t: sample_observable(trajectory, values, t))
    assert np.isnan(compiled(jnp.array([3.0]))).all()
    for index in (-1, 2, True):
        with pytest.raises(ValueError, match="index"):
            radial_velocity(trajectory, index)
    with pytest.raises(ValueError, match="nonzero"):
        radial_velocity(trajectory, line_of_sight=(0, 0, 0))
    with pytest.raises(ValueError, match="parallel"):
        sky_plane_positions(trajectory, up=(0, 0, 1))


def test_bounded_weighted_fit_matches_linear_algebra():
    x = jnp.linspace(-1, 2, 12)
    y = 2*np.asarray(x) + 1 + np.sin(np.arange(12))*0.1
    sigma = np.linspace(0.1, 0.3, 12)
    model = lambda p: p["slope"]*x + p["offset"]
    initial = {"slope": 0.2, "offset": 0.5}
    fit = fit_parameters(model, initial, y, sigma)
    design = np.column_stack([x, np.ones(12)])/sigma[:, None]
    expected = np.linalg.lstsq(design, y/sigma, rcond=None)[0]
    np.testing.assert_allclose(list(fit.parameters.values()), expected, atol=1e-10)
    np.testing.assert_allclose(fit.jacobian, design, atol=1e-14)
    np.testing.assert_allclose(fit.residuals, (fit.prediction-y)/sigma, atol=1e-14)
    assert fit.success and fit.cost < fit.initial_cost
    limited = fit_parameters(model, initial, y, sigma, bounds={"slope": (0, 1)})
    assert limited.success and limited.parameters["slope"] == pytest.approx(1)
    incomplete = fit_parameters(model, initial, y, sigma, max_nfev=1)
    assert not incomplete.success


@pytest.mark.parametrize("method", ["rk4", "leapfrog"])
def test_recovers_planet_mass_and_phase_from_noisy_refined_truth(method, tmp_path):
    from main import radial_velocity_demo
    fit = radial_velocity_demo(method=method, steps=1000, plot=False, output_dir=tmp_path)
    assert fit.success
    assert fit.parameters["planet_mass"] == pytest.approx(0.001, rel=0.01)
    assert fit.parameters["phase"] == pytest.approx(0.7, abs=0.015)
    assert fit.cost < fit.initial_cost/1000
    assert 0.5 < fit.chi_squared/94 < 2
    assert json.loads((tmp_path / "fit.json").read_text())["success"]
    with np.load(tmp_path / "observations.npz") as data:
        assert data["d_rv_d_mass"].shape == data["observations"].shape
        assert np.isfinite(data["d_rv_d_mass"]).all()


def test_invalid_state_configuration_and_unsupported_forces():
    with pytest.raises(ValueError, match="positive"):
        make_state([-1], [[0, 0, 0]], [[0, 0, 0]])
    with pytest.raises(ValueError, match="shape"):
        make_state([1], [[0, 0]], [[0, 0, 0]])
    with pytest.raises(ValueError, match="finite"):
        make_state([1], [[np.nan, 0, 0]], [[0, 0, 0]])
    state = binary_state()
    for extra in ({"steps": -1}, {"steps": True}, {"dt": 0}, {"method": "typo"}):
        with pytest.raises(ValueError):
            simulate(state, **(dict(steps=1, dt=0.1) | extra))
    coincident = state._replace(positions=jnp.zeros((2, 3)))
    with pytest.raises(ValueError, match="Coincident"):
        simulate(coincident, dt=0.1, steps=1)
    assert np.isfinite(simulate(coincident, dt=0.01, steps=1, epsilon=0.1).positions).all()
    for method in ("particle-mesh", "barnes-hut", "multipole"):
        with pytest.raises(ValueError, match="direct"):
            Universe(force_method=method).simulate_differentiable(1, state=state)


@pytest.mark.parametrize("extra,match", [
    ({"uncertainties": 0}, "positive"),
    ({"uncertainties": [1, 2, 3]}, "broadcast"),
    ({"observations": [np.nan, 0]}, "finite"),
    ({"bounds": {"typo": (0, 1)}}, "unknown"),
    ({"bounds": {"a": (2, 3)}}, "contain"),
    ({"max_nfev": 0}, "positive integer"),
])
def test_fit_input_errors(extra, match):
    kwargs = dict(observations=[1, 2], uncertainties=1)
    kwargs.update(extra)
    with pytest.raises(ValueError, match=match):
        fit_parameters(lambda p: p["a"]*jnp.ones(2), {"a": 1.0}, **kwargs)


def test_fit_rejects_nonfinite_models_and_mismatched_shapes():
    with pytest.raises(ValueError, match="non-finite"):
        fit_parameters(lambda p: jnp.array([jnp.nan*p["a"]]), {"a": 1.0}, [1], 1)
    with pytest.raises(ValueError, match="shape"):
        fit_parameters(lambda p: jnp.ones(3)*p["a"], {"a": 1.0}, [1], 1)


def test_optional_dependencies_stay_lazy_and_precision_is_explicit():
    subprocess.run([sys.executable, "-c",
        "import core, universe, observables, inference, main, sys; "
        "assert 'jax' not in sys.modules; assert 'scipy' not in sys.modules; "
        "assert 'matplotlib' not in sys.modules"], check=True)
    subprocess.run([sys.executable, "-c", """
import jax
jax.config.update('jax_enable_x64', False)
from core import make_state
try:
    make_state([1], [[0,0,0]], [[0,0,0]])
except RuntimeError as exc:
    assert 'enable_autodiff' in str(exc)
else:
    raise AssertionError('float32 must not be silently accepted')
"""], check=True)
