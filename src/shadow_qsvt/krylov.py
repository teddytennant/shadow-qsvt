"""Krylov linear algebra for classical shadow QSVT.

Given a Hermitian H, a state psi, and a function f, the state-aware core
builds the moment Hankel matrix, its upper-triangular factor Ur, and the
symmetric tridiagonal Jacobi matrix Jr. The coefficient vector is

    c = f(Jr) e_0,    lambda = Ur^{-1} c,

and the prepared state is sum_j lambda_j H^j |psi>. For an observable O the
Krylov matrix B_{a,b} = <psi| H^a O H^b |psi> gives

    y = lambda^dagger B lambda,

which equals <psi| f(H)^dagger O f(H) |psi> when the Krylov subspace is
H-invariant (or r is large enough to span it) and f is applied exactly on Jr.

Moments g_k = <psi| H^k |psi> are computed by dense matvecs. The paper
estimates them with a Hadamard test; this library uses the exact moments.

Polynomial coefficients are ordered from low degree to high degree, so
[c0, c1, c2] means c0 + c1 x + c2 x^2. A callable is a matrix function of
the Hermitian Jacobi matrix: it is applied to the eigenvalues of Jr.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jnp.ndarray
Coefficients = Union[Sequence[complex], Array]
FunctionSpec = Union[Coefficients, Callable]


@dataclass(frozen=True)
class KrylovFactorization:
    """Factorization of the moment Hankel matrix of size `rank`.

    ur is upper triangular and satisfies Gr = ur^T ur on the numerical
    Krylov subspace. jr is the symmetric tridiagonal Jacobi matrix.
    alphas[k] is the diagonal of jr. betas[k] is the superdiagonal entry
    beta_{k+1} (length rank-1).
    """

    moments: Array
    ur: Array
    jr: Array
    rank: int
    alphas: Array
    betas: Array
    method: str


@dataclass(frozen=True)
class KrylovEstimate:
    """Observable estimates from the state-aware Krylov reduction."""

    rank: int
    lam: Array
    prepared: Array
    values: Array
    ur: Array
    jr: Array
    moments: Array
    method: str


@dataclass(frozen=True)
class ObservableAwareResult:
    """Classical history-Gram reduction of the observable-aware procedure.

    ok is False when the history Gram is singular, which the paper reports
    as failure (s larger than the observable-relevant dimension, or linearly
    dependent histories). values[i] estimates <psi| P(H)^dagger O_i P(H) |psi>.
    """

    values: Array
    lam: Array
    J: Array
    size: int
    ok: bool
    reason: str = ""


def krylov_moments(H, psi, max_power: int) -> Array:
    """Return g_k = <psi| H^k |psi> for k = 0..max_power, by dense matvecs.

    Does not renormalize psi. For Hermitian H the moments are real up to
    roundoff; a negligible imaginary part is dropped.
    """

    if max_power < 0:
        raise ValueError("max_power must be non-negative")
    H = jnp.asarray(H)
    psi = jnp.asarray(psi)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    if psi.ndim != 1 or psi.shape[0] != H.shape[0]:
        raise ValueError("psi must be a vector of length matching H")
    values = []
    vector = psi
    for _ in range(max_power + 1):
        values.append(jnp.vdot(psi, vector))
        vector = H @ vector
    moments = jnp.stack(values)
    scale = 1.0 + float(jnp.max(jnp.abs(moments)))
    if float(jnp.max(jnp.abs(jnp.imag(moments)))) <= 1e-10 * scale:
        return jnp.real(moments).astype(jnp.float64)
    return moments


def hankel_from_moments(moments, size: int, offset: int = 0) -> Array:
    """Hankel matrix G_{a,b} = g_{a+b+offset}."""

    moments = jnp.asarray(moments)
    if moments.shape[0] < 2 * (size - 1) + 1 + offset:
        raise ValueError("not enough moments to build the Hankel matrix")
    index = jnp.arange(size)
    return moments[index[:, None] + index[None, :] + offset]


def factor_from_moments(
    moments,
    r: int,
    *,
    tol: float = 1e-10,
    method: str = "recurrence",
) -> KrylovFactorization:
    """Build Ur and Jr from moments g_0 .. g_{2r-1}.

    method="recurrence" follows the corrected Algorithm 7 identities: the
    superdiagonal and diagonal of each new Ur column come from the moment
    recurrence, interior entries come from the three-term Lanczos relation,
    and a vanishing diagonal stops at the numerical rank. The last Jacobi
    diagonal, which the column loop cannot see when the rank equals r, is
    fixed by the identity Ur^T Jr Ur = Ghat so that Jr matches the Cholesky
    path. The missing u_{-1} term in alpha_0 is taken to be 0.

    method="cholesky" builds the Hankel Gram, takes its upper-triangular
    Cholesky factor (rank-revealing if the Gram is singular), and sets
    Jr = Ur^{-T} Ghat Ur^{-1}.
    """

    if r < 1:
        raise ValueError("r must be at least 1")
    if method not in ("recurrence", "cholesky"):
        raise ValueError("method must be 'recurrence' or 'cholesky'")
    real_moments = _real_moments(moments, r)
    if method == "recurrence":
        ur, jr, alphas, betas, rank = _recurrence_factor(real_moments, r, tol)
    else:
        ur, jr, alphas, betas, rank = _cholesky_factor(real_moments, r, tol)
    return KrylovFactorization(
        moments=jnp.asarray(real_moments[: 2 * r], dtype=jnp.float64),
        ur=ur,
        jr=jr,
        rank=rank,
        alphas=alphas,
        betas=betas,
        method=method,
    )


def matrix_function(J, f: FunctionSpec) -> Array:
    """Apply f to a Hermitian matrix.

    A coefficient sequence is a polynomial, applied by Horner, low degree
    first. A callable is applied to the eigenvalues (Jr is symmetric
    tridiagonal, so eigh is stable). The callable must accept the vector of
    eigenvalues and return one value per eigenvalue.
    """

    J = jnp.asarray(J)
    if callable(f):
        return _apply_callable(J, f)
    return _apply_polynomial(J, f)


def coefficients(ur, jr, f: FunctionSpec):
    """Return (c, lam) with c = f(Jr) e_0 and lam = Ur^{-1} c."""

    transformed = matrix_function(jr, f)
    basis = jnp.zeros(jr.shape[0], dtype=transformed.dtype).at[0].set(1)
    coeff = transformed @ basis
    lam = jnp.linalg.solve(jnp.asarray(ur, dtype=coeff.dtype), coeff)
    return coeff, lam


def prepare_state(H, psi, lam) -> Array:
    """Return sum_j lam[j] H^j |psi>."""

    H = jnp.asarray(H)
    psi = jnp.asarray(psi)
    lam = jnp.asarray(lam)
    dtype = jnp.result_type(H, psi, lam)
    acc = jnp.zeros(psi.shape, dtype=dtype)
    vector = psi.astype(dtype)
    operator = H.astype(dtype)
    for weight in lam:
        acc = acc + weight * vector
        vector = operator @ vector
    return acc


def observable_krylov_matrix(H, psi, observable, size: int) -> Array:
    """B_{a,b} = <psi| H^a O H^b |psi> for a, b = 0..size-1."""

    if size < 1:
        raise ValueError("size must be at least 1")
    H = jnp.asarray(H)
    psi = jnp.asarray(psi)
    observable = jnp.asarray(observable)
    powers = []
    vector = psi
    for _ in range(size):
        powers.append(vector)
        vector = H @ vector
    rows = []
    for left in powers:
        row = [jnp.vdot(left, observable @ right) for right in powers]
        rows.append(jnp.stack(row))
    return jnp.stack(rows)


def quadratic_form(lam, matrix) -> Array:
    """Return lam^dagger matrix lam."""

    lam = jnp.asarray(lam)
    matrix = jnp.asarray(matrix)
    return jnp.vdot(lam, matrix @ lam)


def estimate_observables(
    H,
    psi,
    f: FunctionSpec,
    observables,
    r: int,
    *,
    tol: float = 1e-10,
    method: str = "recurrence",
) -> KrylovEstimate:
    """Estimate y_j = <psi| f(H)^dagger O_j f(H) |psi> from Krylov moments.

    Uses the numerical rank, which may be smaller than r. The prepared state
    is P_r(H)|psi> = sum_j lambda_j H^j |psi>.
    """

    observables = list(observables)
    moments = krylov_moments(H, psi, 2 * r - 1)
    factorization = factor_from_moments(moments, r, tol=tol, method=method)
    if factorization.rank < 1:
        raise ValueError("Krylov subspace is empty (psi has zero norm)")
    _, lam = coefficients(factorization.ur, factorization.jr, f)
    prepared = prepare_state(H, psi, lam)
    values = []
    for observable in observables:
        matrix = observable_krylov_matrix(H, psi, observable, factorization.rank)
        values.append(quadratic_form(lam, matrix))
    return KrylovEstimate(
        rank=factorization.rank,
        lam=lam,
        prepared=prepared,
        values=jnp.stack(values) if values else jnp.zeros((0,), dtype=lam.dtype),
        ur=factorization.ur,
        jr=factorization.jr,
        moments=factorization.moments,
        method=method,
    )


def fill_from_coefficients(H, psi, lam, observables) -> Array:
    """Contract caller-supplied power-basis coefficients against each O_j."""

    values = []
    size = int(jnp.asarray(lam).shape[0])
    for observable in observables:
        matrix = observable_krylov_matrix(H, psi, observable, size)
        values.append(quadratic_form(lam, matrix))
    return jnp.stack(values)


def observable_aware_estimate(
    H,
    psi,
    polynomial: Coefficients,
    observables,
    s: int,
    *,
    tol: float = 1e-10,
) -> ObservableAwareResult:
    """Classical reduction of the observable-aware procedure (Algorithm 6).

    Section 5 builds the observable-relevant quotient of dimension s and
    represents it by the history map. The classical estimator does not need
    the history-state circuit. With Omega = sum_i O_i^2 and exact cross
    moments c_{p,q} = <psi| H^p Omega H^q |psi>,

        G_{a,b} = sum_{ell=0}^{s-1} c_{a+ell, b+ell}
        Ghat_{a,b} = sum_{ell=0}^{s-1} c_{a+ell, b+ell+1}
        J = G^{-1} Ghat
        lam = P(J) e_0
        y_i = lam^dagger B_i lam

    where B_i,a,b = <psi| H^a O_i H^b |psi>. J is similar to a Hermitian
    contraction but is not itself Hermitian, so P is applied by Horner.
    Pass polynomial coefficients, low degree first. If G is singular the
    paper's step returns failure; this function does the same.
    """

    if s < 1:
        raise ValueError("s must be at least 1")
    if callable(polynomial):
        raise TypeError(
            "observable-aware J is not Hermitian; pass polynomial coefficients"
        )
    observables = [jnp.asarray(item) for item in observables]
    if not observables:
        raise ValueError("at least one observable is required")
    H = jnp.asarray(H)
    psi = jnp.asarray(psi)
    omega = jnp.zeros_like(H, dtype=jnp.result_type(H, observables[0]))
    for observable in observables:
        omega = omega + observable @ observable

    # c_{p,q} needs H^{2s-1} |psi>.
    powers = []
    vector = psi
    for _ in range(2 * s):
        powers.append(vector)
        vector = H @ vector

    def cross(left: int, right: int):
        return jnp.vdot(powers[left], omega @ powers[right])

    gram = jnp.zeros((s, s), dtype=jnp.result_type(omega, psi))
    shifted = jnp.zeros_like(gram)
    for a in range(s):
        for b in range(s):
            gram = gram.at[a, b].set(
                sum(cross(a + ell, b + ell) for ell in range(s))
            )
            shifted = shifted.at[a, b].set(
                sum(cross(a + ell, b + ell + 1) for ell in range(s))
            )
    hermitian = 0.5 * (gram + jnp.conj(gram).T)
    spectrum = jnp.linalg.eigvalsh(hermitian)
    scale = 1.0 + float(jnp.linalg.norm(hermitian))
    if float(spectrum[0]) <= tol * scale:
        empty = jnp.zeros((0,), dtype=gram.dtype)
        return ObservableAwareResult(
            values=empty,
            lam=empty,
            J=jnp.zeros((0, 0), dtype=gram.dtype),
            size=0,
            ok=False,
            reason="history Gram is singular",
        )
    jacobi = jnp.linalg.solve(gram, shifted)
    transformed = _apply_polynomial(jacobi, polynomial)
    lam = transformed[:, 0]
    values = fill_from_coefficients(H, psi, lam, observables)
    return ObservableAwareResult(
        values=values,
        lam=lam,
        J=jacobi,
        size=s,
        ok=True,
    )


def _real_moments(moments, r: int) -> list[float]:
    moments = jnp.asarray(moments)
    if moments.ndim != 1 or moments.shape[0] < 2 * r:
        raise ValueError(f"need moments g_0 through g_{2 * r - 1}")
    scale = 1.0 + float(jnp.max(jnp.abs(moments[: 2 * r])))
    if jnp.iscomplexobj(moments):
        imag = float(jnp.max(jnp.abs(jnp.imag(moments[: 2 * r]))))
        if imag > 1e-8 * scale:
            raise ValueError("moments of a Hermitian Hamiltonian must be real")
    return [float(jnp.real(moments[k])) for k in range(2 * r)]


def _alpha_from_column(k: int, column, columns) -> float:
    """alpha_k = u_{k,k+1}/u_{k,k} - u_{k-1,k}/u_{k-1,k-1}, with u_{-1} = 0."""

    alpha = column[k] / columns[k][k]
    if k > 0:
        alpha -= columns[k - 1][k] / columns[k - 1][k - 1]
    return alpha


def _recurrence_factor(moments: list[float], r: int, tol: float):
    """Algorithm 7 column recurrence, corrected against the moment identities."""

    columns = [[0.0 for _ in range(r)] for _ in range(r)]
    columns[0][0] = math.sqrt(max(moments[0], 0.0))
    scale0 = max(1.0, abs(moments[0]))
    if columns[0][0] ** 2 <= tol * scale0:
        empty = jnp.zeros((0, 0), dtype=jnp.float64)
        return empty, empty, empty, empty, 0

    alphas: list[float] = []
    # betas[k] = beta_k, with beta_0 = 0.
    betas = [0.0]
    rank = 1
    stopped_early = False
    for k in range(r - 1):
        column = [0.0 for _ in range(r)]
        # Interior entries of column k+1. These use alpha_0..alpha_{k-1} and
        # beta_1..beta_k, which are already known. Equation (280) in the
        # correctness appendix.
        for i in range(k):
            term = alphas[i] * columns[i][k]
            if i + 1 <= k:
                term += betas[i + 1] * columns[i + 1][k]
            if i - 1 >= 0:
                term += betas[i] * columns[i - 1][k]
            column[i] = term
        # u_{k, k+1} from the moment identity.
        overlap = sum(columns[i][k] * column[i] for i in range(k))
        column[k] = (moments[2 * k + 1] - overlap) / columns[k][k]
        resid = moments[2 * k + 2] - sum(column[i] ** 2 for i in range(k + 1))
        scale = max(1.0, abs(moments[0]), abs(moments[2 * k + 2]))
        if resid <= tol * scale:
            alphas.append(_alpha_from_column(k, column, columns))
            stopped_early = True
            break
        if resid < 0.0:
            raise ValueError("moment Gram is not positive semidefinite")
        column[k + 1] = math.sqrt(resid)
        for i in range(k + 2):
            columns[i][k + 1] = column[i]
        rank = k + 2
        alphas.append(_alpha_from_column(k, column, columns))
        betas.append(column[k + 1] / columns[k][k])

    size = rank
    trimmed = [[columns[i][j] for j in range(size)] for i in range(size)]
    if not stopped_early:
        # The column loop produces alpha_0..alpha_{r-2} and beta_1..beta_{r-1}.
        # alpha_{r-1} is fixed by Ur^T Jr Ur = Ghat, using g_{2r-1}.
        alphas.append(_last_alpha(trimmed, alphas, betas, moments[2 * size - 1]))
    jr = _tridiagonal(alphas, betas[1:size])
    ur = jnp.array(trimmed, dtype=jnp.float64)
    return (
        ur,
        jr,
        jnp.array(alphas, dtype=jnp.float64),
        jnp.array(betas[1:size], dtype=jnp.float64),
        size,
    )


def _last_alpha(columns, alphas, betas, corner: float) -> float:
    """Solve u · (Jr u) = g_{2m-1} for the unknown last diagonal.

    alphas has length m-1 and betas[k] = beta_k with betas[0] = 0.
    """

    size = len(columns)
    last = [columns[i][size - 1] for i in range(size)]
    known = [0.0 for _ in range(size)]
    for i in range(size - 1):
        term = alphas[i] * last[i]
        if i - 1 >= 0:
            term += betas[i] * last[i - 1]
        if i + 1 <= size - 1:
            term += betas[i + 1] * last[i + 1]
        known[i] = term
    if size >= 2:
        known[size - 1] = betas[size - 1] * last[size - 2]
    dot = sum(last[i] * known[i] for i in range(size))
    return (corner - dot) / (last[size - 1] ** 2)


def _tridiagonal(alphas, superdiagonal) -> Array:
    size = len(alphas)
    jr = jnp.diag(jnp.array(alphas, dtype=jnp.float64))
    if size >= 2:
        off = jnp.array(superdiagonal, dtype=jnp.float64)
        jr = jr + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
    return jr


def _cholesky_factor(moments: list[float], r: int, tol: float):
    gram = hankel_from_moments(jnp.array(moments, dtype=jnp.float64), r, offset=0)
    shifted = hankel_from_moments(jnp.array(moments, dtype=jnp.float64), r, offset=1)
    ur, rank = _upper_cholesky(gram, tol)
    if rank < 1:
        empty = jnp.zeros((0, 0), dtype=jnp.float64)
        return empty, empty, empty, empty, 0
    ur = ur[:rank, :rank]
    jr = _jr_from_ghat(ur, shifted[:rank, :rank])
    alphas = jnp.diag(jr)
    betas = jnp.diag(jr, k=1) if rank >= 2 else jnp.zeros((0,), dtype=jnp.float64)
    return ur, jr, alphas, betas, rank


def _upper_cholesky(gram: Array, tol: float):
    """Upper-triangular Cholesky, stopping when a pivot underflows.

    On a full-rank Gram this matches jax.numpy.linalg.cholesky transposed.
    A singular Gram makes that routine return NaN, so the factorization is
    done by the same elimination and truncated at the numerical rank.
    """

    size = int(gram.shape[0])
    factor = jnp.linalg.cholesky(0.5 * (gram + gram.T))
    if not bool(jnp.any(jnp.isnan(factor))):
        pivots = jnp.diag(factor)
        rank = size
        for i in range(size):
            scale = max(1.0, abs(float(gram[i, i])))
            if float(pivots[i]) ** 2 <= tol * scale:
                rank = i
                break
        if rank == size:
            return factor.T, size
    return _rank_revealing_cholesky(gram, tol)


def _rank_revealing_cholesky(gram: Array, tol: float):
    size = int(gram.shape[0])
    entries = [[float(gram[i, j]) for j in range(size)] for i in range(size)]
    columns = [[0.0 for _ in range(size)] for _ in range(size)]
    rank = 0
    for j in range(size):
        for i in range(j):
            acc = sum(columns[t][i] * columns[t][j] for t in range(i))
            columns[i][j] = (entries[i][j] - acc) / columns[i][i]
        acc = sum(columns[t][j] ** 2 for t in range(j))
        resid = entries[j][j] - acc
        scale = max(1.0, abs(entries[j][j]))
        if resid <= tol * scale:
            break
        if resid < 0.0:
            raise ValueError("moment Gram is not positive semidefinite")
        columns[j][j] = math.sqrt(resid)
        rank = j + 1
    ur = jnp.array(columns, dtype=jnp.float64)
    return ur, rank


def _jr_from_ghat(ur: Array, ghat: Array) -> Array:
    """Jr = Ur^{-T} Ghat Ur^{-1}."""

    size = int(ur.shape[0])
    solved = jnp.linalg.solve(ur, jnp.eye(size, dtype=ur.dtype))
    jr = solved.T @ ghat @ solved
    return 0.5 * (jr + jr.T)


def _apply_polynomial(matrix: Array, coeffs: Coefficients) -> Array:
    coeff = jnp.asarray(list(coeffs))
    if coeff.ndim != 1 or coeff.shape[0] < 1:
        raise ValueError("polynomial coefficients must be a non-empty 1-d sequence")
    dtype = jnp.result_type(matrix, coeff)
    matrix = jnp.asarray(matrix, dtype=dtype)
    coeff = coeff.astype(dtype)
    eye = jnp.eye(matrix.shape[0], dtype=dtype)
    acc = coeff[-1] * eye
    for index in range(coeff.shape[0] - 2, -1, -1):
        acc = matrix @ acc + coeff[index] * eye
    return acc


def _apply_callable(matrix: Array, function: Callable) -> Array:
    hermitian = 0.5 * (matrix + jnp.conj(matrix).T)
    evals, evecs = jnp.linalg.eigh(hermitian)
    values = jnp.asarray(function(evals))
    if values.shape != evals.shape:
        raise ValueError("callable must return one value per eigenvalue")
    dtype = jnp.result_type(evecs, values)
    scaled = evecs.astype(dtype) * values.astype(dtype)
    return scaled @ jnp.conj(evecs).T.astype(dtype)
