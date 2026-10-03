"""Classical checks for the shadow QSVT Krylov identities."""

import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from shadow_qsvt import (
    InsufficientData,
    estimate_cross_moments,
    estimate_observables,
    exact_branch_projector,
    factor_from_moments,
    fill_observable_matrix,
    krylov_moments,
    median_of_means,
    observable_aware_estimate,
    observable_krylov_matrix,
    pauli_basis_unitary,
    pauli_shadow_snapshots,
    quadratic_form,
    sample_requirements,
    shadow_expectation,
    single_qubit_inverse_snapshot,
    snapshot_trace,
)
from shadow_qsvt.shadow import _X, _Z


def _dense_function(H, psi, f):
    evals, evecs = jnp.linalg.eigh(0.5 * (H + H.T))
    coeff = evecs.conj().T @ psi
    values = jnp.asarray(f(evals))
    return evecs @ (values * coeff)


def _align_columns(reference, other):
    aligned = jnp.array(other)
    for col in range(aligned.shape[1]):
        if float(aligned[col, col] * reference[col, col]) < 0.0:
            aligned = aligned.at[:, col].multiply(-1.0)
    return aligned


def test_invariant_subspace_matches_dense():
    evals = jnp.array([0.3, -0.7, 1.1, 0.4])
    H = jnp.diag(evals)
    psi = jnp.array([0.6, 0.0, 0.8, 0.0])
    observable = jnp.diag(jnp.array([1.0, -2.0, 0.5, 3.0]))
    identity = jnp.eye(4)
    for function, coeffs in (
        (lambda x: x, [0.0, 1.0]),
        (lambda x: x**2, [0.0, 0.0, 1.0]),
    ):
        exact = function(evals) * psi
        for spec in (function, coeffs):
            estimate = estimate_observables(
                H, psi, spec, [identity, observable], r=2
            )
            assert estimate.rank == 2
            assert jnp.allclose(estimate.prepared, exact, atol=1e-6)
            norm_sq = jnp.vdot(estimate.prepared, estimate.prepared)
            assert abs(estimate.values[0] - norm_sq) < 1e-6
            dense_y = jnp.vdot(exact, observable @ exact)
            assert abs(estimate.values[1] - dense_y) < 1e-6


def test_recurrence_matches_cholesky_on_full_rank():
    H = jnp.array(
        [
            [0.4, 0.1, -0.2],
            [0.1, -0.3, 0.15],
            [-0.2, 0.15, 0.6],
        ]
    )
    psi = jnp.array([0.3, -0.5, 0.8])
    psi = psi / jnp.linalg.norm(psi)
    moments = krylov_moments(H, psi, 5)
    recurrence = factor_from_moments(moments, 3, method="recurrence")
    cholesky = factor_from_moments(moments, 3, method="cholesky")
    assert recurrence.rank == 3
    assert cholesky.rank == 3
    aligned = _align_columns(cholesky.ur, recurrence.ur)
    assert jnp.allclose(aligned, cholesky.ur, atol=1e-8)
    assert jnp.allclose(recurrence.jr, cholesky.jr, atol=1e-8)

    exact = _dense_function(H, psi, lambda x: x**2)
    for method in ("recurrence", "cholesky"):
        estimate = estimate_observables(
            H, psi, lambda x: x**2, [jnp.eye(3)], r=3, method=method
        )
        assert jnp.allclose(estimate.prepared, exact, atol=1e-6)


def test_rank_deficient_eigenvector():
    eigenvalue = -0.7
    H = jnp.diag(jnp.array([0.3, eigenvalue, 1.1, 0.4]))
    psi = jnp.array([0.0, 1.0, 0.0, 0.0])
    for function, expected_scale in (
        (lambda x: x, eigenvalue),
        (lambda x: x**2, eigenvalue**2),
        ([0.0, 0.0, 1.0], eigenvalue**2),
    ):
        estimate = estimate_observables(H, psi, function, [jnp.eye(4)], r=4)
        assert estimate.rank == 1
        assert jnp.allclose(estimate.prepared, expected_scale * psi, atol=1e-6)
        assert abs(estimate.values[0] - expected_scale**2) < 1e-6


def test_median_of_means_and_observable_matrix():
    constant = 3.5
    got = median_of_means(jnp.full((100,), constant), 4, 10)
    assert float(got) == constant

    groups, group_size, need = sample_requirements(1, 0.1, 1.0, 1.0)
    short = median_of_means(jnp.ones(need - 1), groups, group_size)
    assert isinstance(short, InsufficientData)
    assert short.have == need - 1
    assert short.need == need

    dataset = [(1.0, jnp.eye(2) / 2) for _ in range(need - 1)]
    rejected = estimate_cross_moments(
        dataset, [jnp.eye(2)], nu2=1.0, eps=1.0, eta=0.1
    )
    assert isinstance(rejected, InsufficientData)
    assert rejected.need == need

    evals = jnp.array([0.3, -0.7, 1.1, 0.4])
    H = jnp.diag(evals)
    psi = jnp.array([0.6, 0.0, 0.8, 0.0])
    observable = jnp.diag(jnp.array([1.0, -2.0, 0.5, 3.0]))
    estimate = estimate_observables(H, psi, lambda x: x**2, [observable], r=2)
    exact_mu = observable_krylov_matrix(H, psi, observable, estimate.rank)
    filled = fill_observable_matrix(exact_mu)
    contracted = quadratic_form(estimate.lam, filled)
    exact_state = (evals**2) * psi
    dense = jnp.vdot(exact_state, observable @ exact_state)
    assert abs(contracted - dense) < 1e-6


def test_exact_rho_and_pauli_shadows():
    phi = jnp.array([0.6, 0.8j], dtype=jnp.complex128)
    phi = phi / jnp.linalg.norm(phi)
    rho = jnp.outer(phi, jnp.conj(phi))
    pauli = _Z
    got = snapshot_trace(1.0, rho, pauli)
    exact = jnp.vdot(phi, pauli @ phi)
    assert abs(got - exact) < 1e-12
    mean = shadow_expectation([(1.0, rho)], pauli)
    assert abs(mean - exact) < 1e-12

    # Enumerate the six local Pauli outcomes. The inverse channel is unbiased.
    plus = jnp.array([1.0, 1.0], dtype=jnp.complex128) / math.sqrt(2.0)
    projector = jnp.outer(plus, jnp.conj(plus))
    recovered = jnp.zeros((2, 2), dtype=jnp.complex128)
    for basis in range(3):
        unitary = pauli_basis_unitary(basis)
        rotated = unitary @ plus
        for bit in range(2):
            weight = jnp.abs(rotated[bit]) ** 2 / 3.0
            recovered = recovered + weight * single_qubit_inverse_snapshot(basis, bit)
    assert jnp.allclose(recovered, projector, atol=1e-12)

    snapshots = pauli_shadow_snapshots(plus, 4000, key=0)
    monte_carlo = shadow_expectation(snapshots, _X)
    assert abs(monte_carlo - 1.0) < 0.2


def test_exact_branch_recovers_cross_moment():
    H = jnp.diag(jnp.array([0.2, -0.5]))
    psi = jnp.array([0.6, 0.8])
    observable = jnp.array([[0.3, 0.1], [0.1, -0.4]])
    phi_a = jnp.linalg.matrix_power(H, 2) @ psi
    phi_b = H @ psi
    mu = jnp.vdot(phi_a, observable @ phi_b)
    rho = exact_branch_projector(phi_a, phi_b)
    _, _, need = sample_requirements(1, 0.2, 1.0, 2.0)
    dataset = [(1.0, rho) for _ in range(need)]
    estimated = estimate_cross_moments(
        dataset, [observable], nu2=1.0, eps=2.0, eta=0.2
    )
    assert abs(estimated[0] - mu) < 1e-10
    assert abs(mu) > 1e-3


def test_observable_aware_matches_dense_4x4():
    # Reduced quotient: only the first eigenvector is visible, s = 1.
    H = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
    psi = jnp.ones(4) / 2.0
    visible = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
    reduced = observable_aware_estimate(H, psi, [0.0, 0.0, 1.0], [visible], s=1)
    assert reduced.ok
    exact_reduced = abs(1.0**2 * psi[0]) ** 2
    assert abs(reduced.values[0] - exact_reduced) < 1e-8

    # Two observables, s equal to the visible dimension, degree above s-1.
    first = jnp.diag(jnp.array([1.0, 0.0, 0.0, 0.0]))
    second = jnp.diag(jnp.array([0.0, 1.0, 0.0, 0.0]))
    coeffs = [0.5, -1.0, 0.3]
    pair = observable_aware_estimate(H, psi, coeffs, [first, second], s=2)
    assert pair.ok

    def poly(x):
        return coeffs[0] + coeffs[1] * x + coeffs[2] * x**2

    fpsi = poly(jnp.diag(H)) * psi
    for index, observable in enumerate((first, second)):
        dense = jnp.vdot(fpsi, observable @ fpsi)
        assert abs(pair.values[index] - dense) < 1e-8

    # Generic 4x4 Hermitian, full observable-relevant dimension.
    generic_H = jnp.array(
        [
            [0.5, 0.1, 0.0, -0.2],
            [0.1, -0.3, 0.2, 0.0],
            [0.0, 0.2, 0.4, 0.1],
            [-0.2, 0.0, 0.1, -0.1],
        ]
    )
    generic_psi = jnp.array([0.5, -0.3, 0.2, 0.7])
    generic_psi = generic_psi / jnp.linalg.norm(generic_psi)
    generic_O = jnp.array(
        [
            [0.2, 0.0, 0.1, 0.0],
            [0.0, -0.4, 0.0, 0.2],
            [0.1, 0.0, 0.3, 0.0],
            [0.0, 0.2, 0.0, 0.1],
        ]
    )
    cubic = [0.2, -0.5, 0.1, 0.05]
    aware = observable_aware_estimate(
        generic_H, generic_psi, cubic, [generic_O], s=4
    )
    assert aware.ok
    dense_state = _dense_function(
        generic_H,
        generic_psi,
        lambda x: cubic[0] + cubic[1] * x + cubic[2] * x**2 + cubic[3] * x**3,
    )
    dense_y = jnp.vdot(dense_state, generic_O @ dense_state)
    assert abs(aware.values[0] - dense_y) < 1e-8
