"""Numerical regressions for the independent force and integration choices."""

import numpy as np
import pytest

from core import Body, MultipoleExpansion, ParticleMesh, build_octree
from universe import Universe


def binary(dt=0.01, **kwargs):
    u = Universe(dt=dt, G=1, epsilon=0, **kwargs)
    u.add_body(Body("a", 0.5, [-0.5, 0, 0], [0, -0.5, 0]))
    u.add_body(Body("b", 0.5, [0.5, 0, 0], [0, 0.5, 0]))
    return u


def cloud(method="direct", theta=0.5, epsilon=0.1):
    rng = np.random.default_rng(12)
    u = Universe(G=1, epsilon=epsilon, force_method=method, theta=theta,
                 grid_size=16, box_size=8)
    for i in range(40):
        u.add_body(Body(str(i), rng.uniform(0.3, 2), rng.normal(size=3), [0, 0, 0]))
    return u


def accelerations(u):
    u.compute_accelerations()
    return np.array([b.acceleration for b in u.bodies])


@pytest.mark.parametrize("integrator,ratio", [("rk4", 16), ("leapfrog", 4)])
def test_orbit_convergence(integrator, ratio):
    errors = []
    for steps in (20, 40):
        u = binary(dt=1/steps, record_every=0)
        u.run(steps, method=integrator, progress=False)
        exact = 0.5 * np.array([np.cos(1), np.sin(1), 0])
        errors.append(np.linalg.norm(u.bodies[1].position - exact))
    assert ratio * 0.9 < errors[0] / errors[1] < ratio * 1.1


def test_leapfrog_energy_and_momentum():
    u = binary(dt=0.02)
    u.run(1500, progress=False)
    energy = np.array(u.diag_E)
    assert np.max(abs((energy - energy[0]) / energy[0])) < 1e-6
    np.testing.assert_allclose(u.diag_P, 0, atol=1e-14)
    np.testing.assert_allclose(np.array(u.diag_L) - u.diag_L[0], 0, atol=1e-14)


@pytest.mark.parametrize("method", Universe.FORCE_METHODS)
@pytest.mark.parametrize("integrator", Universe.INTEGRATORS)
def test_all_combinations_and_final_acceleration(method, integrator):
    u = binary(force_method=method, grid_size=16, box_size=8)
    u.run(3, method=integrator, progress=False)
    assert u.time == pytest.approx(0.03)
    assert u.diag_time == pytest.approx([0, 0.01, 0.02, 0.03])
    assert all(len(b.history_pos) == 4 for b in u.bodies)
    assert np.isfinite(u.diag_E).all()
    before = np.array([b.acceleration.copy() for b in u.bodies])
    np.testing.assert_allclose(before, accelerations(u), rtol=1e-14, atol=1e-14)
    saved = u.bodies[0].history_pos[-1].copy()
    u.bodies[0].position += 1
    np.testing.assert_array_equal(u.bodies[0].history_pos[-1], saved)


def test_force_matches_gradient_of_softened_energy():
    u = cloud()
    acc = accelerations(u)
    b = u.bodies[3]
    for axis in range(3):
        x0 = b.position[axis]
        b.position[axis] = x0 + 1e-5
        plus = u.potential_energy()
        b.position[axis] = x0 - 1e-5
        minus = u.potential_energy()
        b.position[axis] = x0
        assert -(plus - minus) / 2e-5 / b.mass == pytest.approx(acc[3, axis], rel=1e-8)
    np.testing.assert_allclose(sum(b.mass*b.acceleration for b in u.bodies), 0, atol=1e-12)


@pytest.mark.parametrize("method", ["barnes-hut", "multipole"])
def test_tree_exact_limit(method):
    np.testing.assert_allclose(accelerations(cloud(method, theta=0)),
                               accelerations(cloud()), rtol=1e-13, atol=1e-13)


def test_tree_moments_and_quadrupole_accuracy():
    u = cloud()
    tree = build_octree(u.bodies)
    mass = sum(b.mass for b in u.bodies)
    com = sum(b.mass * b.position for b in u.bodies) / mass
    moment = sum(b.mass * np.outer(b.position - com, b.position - com) for b in u.bodies)
    np.testing.assert_allclose(tree.com, com, atol=1e-14)
    np.testing.assert_allclose(tree.moment, moment, atol=1e-13)
    assert np.trace(tree.quadrupole) == pytest.approx(0, abs=1e-12)
    exact = accelerations(u)
    mono_error = np.linalg.norm(accelerations(cloud("barnes-hut", theta=0.7)) - exact)
    quad_error = np.linalg.norm(accelerations(cloud("multipole", theta=0.7)) - exact)
    assert quad_error < mono_error * 0.5


@pytest.mark.parametrize("epsilon", [0, 2.0])
def test_quadrupole_far_field_expands_softened_kernel(epsilon):
    sources = [Body("a", 2, [-0.3, 0.1, 0], [0, 0, 0]),
               Body("b", 1, [0.6, -0.2, 0], [0, 0, 0])]
    target = Body("target", 1, [10, 8, 4], [0, 0, 0])
    tree = build_octree(sources)
    exact = sum(b.mass*(b.position-target.position) /
                (np.sum((b.position-target.position)**2)+epsilon**2)**1.5 for b in sources)
    mono = tree.compute_acceleration(target, 1, 1, epsilon, order=0)
    quad = tree.compute_acceleration(target, 1, 1, epsilon, order=2)
    assert np.linalg.norm(quad-exact) < 0.05*np.linalg.norm(mono-exact)


@pytest.mark.parametrize("method", ["direct", "barnes-hut", "multipole"])
def test_coincident_particles_and_no_self_force(method):
    u = Universe(G=1, epsilon=0.2, force_method=method, theta=100)
    for i in range(4):
        u.add_body(Body(str(i), 1, [0, 0, 0], [0, 0, 0]))
    np.testing.assert_array_equal(accelerations(u), np.zeros((4, 3)))
    assert np.isfinite(u.potential_energy())
    u.epsilon = 0
    with pytest.raises(ValueError, match="Coincident"):
        u.compute_accelerations()
    two = binary(force_method=method, theta=100)
    np.testing.assert_allclose(accelerations(two), [[0.5, 0, 0], [-0.5, 0, 0]])


def test_pm_periodic_translation_and_momentum():
    u = cloud("particle-mesh")
    before = accelerations(u)
    energy = u.potential_energy()
    for i, b in enumerate(u.bodies):
        b.position += np.array([i-3, -2*i, 1]) * 8
    np.testing.assert_allclose(accelerations(u), before, atol=5e-12)
    assert u.potential_energy() == pytest.approx(energy, rel=1e-13)
    np.testing.assert_allclose(sum(b.mass*b.acceleration for b in u.bodies), 0, atol=2e-12)


def test_pm_cic_mass_self_force_and_fixed_fit():
    u = cloud("particle-mesh")
    rho, _, _ = u.particle_mesh._solve(u.bodies, u.G)
    assert rho.sum() * (8/16)**3 == pytest.approx(sum(b.mass for b in u.bodies))
    pm = ParticleMesh(grid_size=16)
    b = Body("single", 2, [0.31, -0.26, 0.73], [0, 0, 0])
    pm.compute_accelerations([b], 1)
    size, origin = pm.box_size, pm.origin.copy()
    b.position += [0.234, -5.52, 1.399]
    pm.compute_accelerations([b], 1)
    np.testing.assert_allclose(b.acceleration, 0, atol=1e-12)
    assert pm.box_size == size
    np.testing.assert_array_equal(pm.origin, origin)


@pytest.mark.parametrize("n", [15, 16])
def test_pm_analytic_poisson_mode(n):
    pm = ParticleMesh(n, box_size=2*np.pi)
    x = np.arange(n) * (2*np.pi/n)
    rho = np.broadcast_to(2 + np.cos(x)[:, None, None], (n, n, n)).copy()
    phi = pm._solve_density(rho, G=1)
    expected = np.broadcast_to(-4*np.pi*np.cos(x)[:, None, None], rho.shape)
    np.testing.assert_allclose(phi, expected, atol=1e-13)
    np.testing.assert_allclose(pm._solve_density(np.ones_like(rho), G=1), 0, atol=1e-13)


@pytest.mark.parametrize("method", Universe.FORCE_METHODS)
@pytest.mark.parametrize("integrator", Universe.INTEGRATORS)
def test_empty_and_single_body(method, integrator):
    empty = Universe(force_method=method, grid_size=8, box_size=10)
    empty.run(1, method=integrator, progress=False)
    np.testing.assert_array_equal(empty.linear_momentum(), np.zeros(3))
    assert empty.total_energy() == 0
    empty.add_body(Body("single", 1, [0, 0, 0], [1, 2, 3]))
    empty.run(1, method=integrator, progress=False)
    np.testing.assert_allclose(empty.bodies[0].position, [0.1, 0.2, 0.3], atol=1e-14)


def test_recording_cadence_and_continuation():
    u = binary(record_every=3)
    u.run(5, progress=False)
    u.run(2, progress=False)
    assert u.diag_time == pytest.approx([0, 0.03, 0.05, 0.06, 0.07])
    assert len(u.bodies[0].history_pos) == 5
    u = binary(record_every=0)
    u.run(2, progress=False)
    assert u.diag_time == u.bodies[0].history_pos == []


@pytest.mark.parametrize("kwargs", [{"dt": 0}, {"G": -1}, {"epsilon": -1},
    {"theta": np.nan}, {"force_method": "typo"}, {"multipole_order": 1},
    {"grid_size": 2}, {"grid_size": 8.5}, {"box_size": 0}, {"record_every": -1}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Universe(**kwargs)


def test_invalid_state_and_run():
    u = binary()
    for value in (-1, 2.5, True):
        with pytest.raises(ValueError):
            u.run(value)
    with pytest.raises(ValueError, match="integrator"):
        u.run(1, method="typo")
    with pytest.raises(ValueError):
        u.add_body(u.bodies[0])
    for mass in (0, -1, np.inf):
        with pytest.raises(ValueError):
            Body("bad", mass, [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        Body("bad", 1, [0, 0], [0, 0, 0])
    vector = np.ones(3)
    body = Body("copy", 1, vector, vector)
    body.position[0] = 7
    np.testing.assert_array_equal(vector, np.ones(3))
    np.testing.assert_array_equal(body.velocity, np.ones(3))


def test_legacy_attachment_uses_universe_dispatch():
    from stub import FastMultipole, attach
    u = binary()
    attach(u, FastMultipole(theta=0))
    assert u.compute_accelerations.__func__ is Universe.compute_accelerations
    np.testing.assert_allclose(accelerations(u), accelerations(binary()))
    attach(u, ParticleMesh(grid_size=8, box_size=8))
    u.run(2, method="rk4", progress=False)
    assert u.force_method == "particle-mesh"


def test_pm_full_pipeline_force_and_mesh_convergence():
    errors = []
    for n in (8, 16):
        h = 2*np.pi/n
        positions = np.array(np.meshgrid(*([np.arange(n)*h]*3), indexing="ij")).reshape(3, -1).T
        u = Universe(G=1, force_method="particle-mesh", grid_size=n,
                     box_size=2*np.pi, box_origin=[0, 0, 0])
        # Grid-aligned particles deposit the known density rho = 2 + cos(x).
        for i, pos in enumerate(positions):
            u.add_body(Body(str(i), (2+np.cos(pos[0]))*h**3, pos, [0, 0, 0]))
        acc = accelerations(u)
        exact_x = -4*np.pi*np.sin(positions[:, 0])
        np.testing.assert_allclose(acc[:, 0], exact_x*np.sin(h)/h, atol=1e-12)
        np.testing.assert_allclose(acc[:, 1:], 0, atol=1e-12)
        errors.append(np.linalg.norm(acc[:, 0]-exact_x) / np.sqrt(len(positions)))
    assert 3.8 < errors[0]/errors[1] < 4.1


def test_pm_boundary_crossing_attracts_across_seam():
    u = Universe(G=1, force_method="particle-mesh", box_size=8, grid_size=32)
    u.add_body(Body("left", 1, [-3.75, 0, 0], [0, 0, 0]))
    u.add_body(Body("right", 1, [3.75, 0, 0], [0, 0, 0]))
    acc = accelerations(u)
    assert acc[0, 0] < 0 < acc[1, 0]
    np.testing.assert_allclose(acc[0], -acc[1], atol=1e-13)


def test_headless_imports_do_not_load_plotting():
    import subprocess
    import sys
    subprocess.run([sys.executable, "-c",
                    "import core, universe, main, generate_results, sys; "
                    "assert 'matplotlib' not in sys.modules"], check=True)
