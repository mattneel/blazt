//! Public error sets used by BLAS/LAPACK-style routines.
//!
//! These are re-exported from `blazt` root for ergonomic use, e.g. `blazt.LuError`.

/// LU factorization errors.
pub const LuError = error{ Singular };

/// Cholesky factorization errors.
pub const CholeskyError = error{ NotPositiveDefinite };

/// Triangular solve (vector) errors.
pub const TrsvError = error{ Singular };

/// Triangular solve (matrix) errors.
pub const TrsmError = error{ Singular };

/// Singular Value Decomposition errors.
pub const SvdError = error{ NoConvergence };

/// Eigen decomposition errors.
pub const EigError = error{ NoConvergence };


