"""Classical simulation of shadow QSVT observable estimates."""

import jax

jax.config.update("jax_enable_x64", True)

from shadow_qsvt.krylov import (
    KrylovEstimate,
    KrylovFactorization,
    ObservableAwareResult,
    estimate_observables,
    factor_from_moments,
    fill_from_coefficients,
    krylov_moments,
    matrix_function,
    observable_aware_estimate,
    observable_krylov_matrix,
    prepare_state,
    quadratic_form,
)
from shadow_qsvt.shadow import (
    InsufficientData,
    estimate_cross_moments,
    exact_branch_projector,
    fill_observable_matrix,
    median_of_means,
    pauli_basis_unitary,
    pauli_shadow_snapshots,
    sample_requirements,
    shadow_expectation,
    single_qubit_inverse_snapshot,
    snapshot_trace,
)

__all__ = [
    "InsufficientData",
    "KrylovEstimate",
    "KrylovFactorization",
    "ObservableAwareResult",
    "estimate_cross_moments",
    "estimate_observables",
    "exact_branch_projector",
    "factor_from_moments",
    "fill_from_coefficients",
    "fill_observable_matrix",
    "krylov_moments",
    "matrix_function",
    "median_of_means",
    "observable_aware_estimate",
    "observable_krylov_matrix",
    "pauli_basis_unitary",
    "pauli_shadow_snapshots",
    "prepare_state",
    "quadratic_form",
    "sample_requirements",
    "shadow_expectation",
    "single_qubit_inverse_snapshot",
    "snapshot_trace",
]

__version__ = "0.1.0"
