"""Classical half of the shadow post-processing (Algorithms 4, 5, and 8).

Snapshots are caller-supplied. On a classical simulation a snapshot may be
the exact projector, or a Pauli-shadow reconstruction of a register small
enough to store as a matrix. There is no block-encoding circuit and no
fault-tolerant measurement schedule.

Algorithm 4, for a dataset of snapshots (s_t, rho_t) on the control plus
system register and a system observable O:

    R_t = s_t * Tr[(X_C tensor O) rho_t]
    I_t = s_t * Tr[(Y_C tensor O) rho_t]
    mu = median_of_means(R) + i * median_of_means(I)

The control is the leading subsystem, so X_C tensor O is kron(X, O).
The success bit s_t is 1 when a block-encoding ancilla would have measured
0. The system-only Pauli sampler has no ancilla, so every shot has weight 1.

Median-of-means uses K = ceil(8 ln(2M / eta)) and S = ceil(8 nu^2 / eps^2).
The logarithm is natural: the factor 8 matches a Hoeffding bound whose
failure probability decays as exp(-K/8). A dataset shorter than Nreq = K*S
is rejected. It is not truncated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray

_I = jnp.eye(2, dtype=jnp.complex128)
_X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
_Y = jnp.array([[0.0, -1j], [1j, 0.0]], dtype=jnp.complex128)
_Z = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
_H = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=jnp.complex128) / math.sqrt(2)
_SDG = jnp.array([[1.0, 0.0], [0.0, -1j]], dtype=jnp.complex128)
_PAULI = (_X, _Y, _Z)


@dataclass(frozen=True)
class InsufficientData:
    """The snapshot dataset is shorter than the median-of-means requirement."""

    have: int
    need: int


def sample_requirements(num_observables: int, eta: float, nu2: float, eps: float):
    """Return (K, S, Nreq) from Algorithm 4.

    K = ceil(8 ln(2M / eta)), S = ceil(8 nu^2 / eps^2), Nreq = K * S.
    """

    if num_observables < 1:
        raise ValueError("M must be positive")
    if not (0.0 < eta < 1.0):
        raise ValueError("eta must lie in (0, 1)")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if nu2 < 0.0:
        raise ValueError("nu2 must be non-negative")
    groups = math.ceil(8.0 * math.log(2.0 * num_observables / eta))
    group_size = math.ceil(8.0 * nu2 / (eps ** 2))
    if groups < 1 or group_size < 1:
        raise ValueError("median-of-means sizes must be positive")
    return groups, group_size, groups * group_size


def median_of_means(samples, K: int, S: int):
    """Median of K group means of length S.

    Uses the first K*S samples. If the array is shorter than K*S, returns
    InsufficientData and does not truncate. Complex samples are split into
    real and imaginary parts, matching the paper's separate estimators.
    """

    samples = jnp.asarray(samples)
    if samples.ndim != 1:
        raise ValueError("samples must be a 1-d array")
    groups = int(K)
    group_size = int(S)
    if groups < 1 or group_size < 1:
        raise ValueError("K and S must be positive")
    if jnp.iscomplexobj(samples):
        real = median_of_means(jnp.real(samples), groups, group_size)
        imag = median_of_means(jnp.imag(samples), groups, group_size)
        if isinstance(real, InsufficientData):
            return real
        return real + 1j * imag
    need = groups * group_size
    have = int(samples.shape[0])
    if have < need:
        return InsufficientData(have, need)
    used = samples[:need].reshape((groups, group_size))
    return jnp.median(jnp.mean(used, axis=1))


def snapshot_trace(success, rho, observable) -> Array:
    """Return s * Tr(observable @ rho).

    For s = 1 and rho = |phi><phi|, this is <phi| observable |phi>.
    """

    rho = jnp.asarray(rho)
    observable = jnp.asarray(observable)
    return success * jnp.trace(observable @ rho)


def shadow_expectation(snapshots, observable) -> Array:
    """Sample mean of s_t * Tr(observable @ rho_t).

    Each snapshot is either a density (or shadow) matrix, with success weight
    1, or a pair (s_t, rho_t). This is the unbiased classical-shadow mean.
    Algorithm 4's robust estimator is estimate_cross_moments, which uses
    median of means on the control-system traces.
    """

    observable = jnp.asarray(observable)
    if len(snapshots) < 1:
        raise ValueError("snapshots must be non-empty")
    total = 0.0
    for item in snapshots:
        success, rho = _split_snapshot(item)
        total = total + snapshot_trace(success, rho, observable)
    return total / len(snapshots)


def estimate_cross_moments(dataset, observables, nu2: float, eps: float, eta: float):
    """Algorithm 4 for one pair of Krylov powers.

    dataset is a sequence of (s_t, rho_t) with rho_t on C+S (control leading).
    Returns a complex array mu_i, one per observable, or InsufficientData if
    the dataset is shorter than Nreq. Does not silently drop samples below
    Nreq. Extra samples beyond Nreq are ignored (any Nreq samples suffice).
    """

    observables = [jnp.asarray(item) for item in observables]
    groups, group_size, need = sample_requirements(
        len(observables), eta, nu2, eps
    )
    have = len(dataset)
    if have < need:
        return InsufficientData(have, need)
    selected = dataset[:need]
    estimates = []
    for observable in observables:
        real_obs = jnp.kron(_X, observable)
        imag_obs = jnp.kron(_Y, observable)
        real_samples = []
        imag_samples = []
        for success, rho in selected:
            real_samples.append(jnp.real(snapshot_trace(success, rho, real_obs)))
            imag_samples.append(jnp.real(snapshot_trace(success, rho, imag_obs)))
        real_est = median_of_means(jnp.stack(real_samples), groups, group_size)
        imag_est = median_of_means(jnp.stack(imag_samples), groups, group_size)
        estimates.append(real_est + 1j * imag_est)
    return jnp.stack(estimates)


def fill_observable_matrix(mu) -> Array:
    """Algorithm 5, steps 3-4.

    Diagonal entries use the real part. For a < b, B_{a,b} = mu_{a,b} and
    B_{b,a} = conj(mu_{a,b}). Entries of mu below the diagonal are ignored,
    because the shadow procedure estimates each unordered pair once.
    """

    mu = jnp.asarray(mu)
    if mu.ndim != 2 or mu.shape[0] != mu.shape[1]:
        raise ValueError("mu must be square")
    upper = jnp.triu(mu, k=1)
    diagonal = jnp.real(jnp.diag(mu))
    return jnp.diag(diagonal) + upper + jnp.conj(upper).T


def exact_branch_projector(phi_a, phi_b) -> Array:
    """Projector of |chi> = 1/sqrt(2) (|0>|phi_a> + |1>|phi_b>).

    Control is the leading subsystem, matching kron(X, O). Feeding this
    projector with s = 1 into estimate_cross_moments recovers
    <phi_a| O |phi_b> with zero statistical error.
    """

    phi_a = jnp.asarray(phi_a)
    phi_b = jnp.asarray(phi_b)
    if phi_a.shape != phi_b.shape:
        raise ValueError("branch states must have the same shape")
    chi = jnp.concatenate([phi_a, phi_b]) / math.sqrt(2.0)
    return jnp.outer(chi, jnp.conj(chi))


def pauli_basis_unitary(basis: int) -> Array:
    """Unitary U with U^dagger Z U equal to X, Y, or Z.

    Applying U and measuring Z measures that Pauli. basis 0, 1, 2 are X, Y, Z.
    """

    if basis == 0:
        return _H
    if basis == 1:
        return _H @ _SDG
    if basis == 2:
        return _I
    raise ValueError("basis must be 0 (X), 1 (Y), or 2 (Z)")


def single_qubit_inverse_snapshot(basis: int, bit: int) -> Array:
    """Standard single-qubit inverse channel, 3 U^dagger |b><b| U - I.

    Equals I/2 + (3/2) (-1)^b P, as in the local Pauli specialization.
    """

    outcome = 1.0 - 2.0 * int(bit)
    return 0.5 * _I + 1.5 * outcome * _PAULI[int(basis)]


def pauli_shadow_snapshots(state, n_shots: int, key) -> list[Array]:
    """Sample local Pauli classical shadows of a pure state.

    state has shape (2^n,). Qubit 0 is the least significant bit. Each
    snapshot is the tensor product of single-qubit inverse-channel matrices
    3 U_j^dagger |b_j><b_j| U_j - I. Intended for n <= 3. The returned
    matrices are not states; their expectation equals the true projector.
    """

    state = jnp.asarray(state, dtype=jnp.complex128)
    if state.ndim != 1:
        raise ValueError("state must be a vector")
    dim = int(state.shape[0])
    n_qubits = int(round(math.log2(dim))) if dim > 0 else -1
    if n_qubits < 1 or 2 ** n_qubits != dim:
        raise ValueError("state dimension must be a power of two")
    if n_qubits > 8:
        raise ValueError("this classical sampler is only for small registers")
    if n_shots < 1:
        raise ValueError("n_shots must be positive")
    key = jax.random.PRNGKey(int(key)) if isinstance(key, int) else key
    snapshots = []
    for _ in range(n_shots):
        key, basis_key, outcome_key = jax.random.split(key, 3)
        bases = jax.random.randint(basis_key, (n_qubits,), 0, 3)
        unitaries = [pauli_basis_unitary(int(bases[q])) for q in range(n_qubits)]
        rotated = _kron_lsb(unitaries) @ state
        probs = jnp.real(jnp.abs(rotated) ** 2)
        probs = probs / jnp.sum(probs)
        index = int(jax.random.choice(outcome_key, dim, p=probs))
        factors = []
        for qubit in range(n_qubits):
            bit = (index >> qubit) & 1
            factors.append(single_qubit_inverse_snapshot(int(bases[qubit]), bit))
        snapshots.append(_kron_lsb(factors))
    return snapshots


def _kron_lsb(ops) -> Array:
    """Kronecker product with ops[0] on the least significant qubit."""

    out = ops[0]
    for op in ops[1:]:
        out = jnp.kron(op, out)
    return out


def _split_snapshot(item):
    if isinstance(item, tuple) and len(item) == 2:
        return item[0], item[1]
    return 1.0, item
