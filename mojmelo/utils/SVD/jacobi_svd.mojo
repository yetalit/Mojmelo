# ==============================================================================
# Based on libeigen — Eigen/src/SVD/JacobiSVD.h, Copyright (C) 2009-2010 Benoit
# Jacob, Copyright (C) 2013-2014 Gael Guennebaud, MPL-2.0.
# ------------------------------------------------------------------------------
# Eigen's JacobiSVD reduces a rectangular m x p matrix to a square diagSize x
# diagSize problem (diagSize = min(m, p)) via a QR decomposition first (the
# "R-SVD" step), then runs the two-sided Jacobi sweep on that square work
# matrix.
#
#   * `householder_qr(...)`      — plain (non-pivoting) Householder QR,
#                                   standing in for Eigen's
#                                   HouseholderQRPreconditioner. Eigen also
#                                   offers ColPivHouseholderQR (the default)
#                                   and FullPivHouseholderQR preconditioners,
#                                   which are more numerically robust for
#                                   rank-deficient/ill-conditioned inputs but
#                                   require column-pivoting machinery this
#                                   port doesn't have; not ported.
#
# NOT ported:
#   * Thin U/V — this port always produces full square U (rows x rows) and
#     full square V (cols x cols), matching what JacobiSVD's ComputeFullU |
#     ComputeFullV options would result.
#
# CONVERGENCE: capped at `max_sweeps` cyclic sweeps over all (p,q) pairs;
# returns INFO_NO_CONVERGENCE (not a crash) if that cap is hit without every
# off-diagonal pair falling under the threshold — mirrors Eigen's own
# NoConvergence outcome (Eigen itself has no such cap and simply loops until
# converged; the cap here is a deliberate safety valve).
# ==============================================================================

from .linalg_core import (
    RealScalar,
    REAL_EPSILON,
    REAL_MIN,
    Vec,
    Mat,
    ComputationInfo,
    INFO_SUCCESS,
    INFO_NO_CONVERGENCE,
    mat_identity,
    mat_transpose,
)
from .jacobi import (
    jacobi_svd_2x2,
    apply_rotation_left_rows,
    apply_rotation_right_cols,
    apply_rotation_cols_direct,
)

# ------------------------------------------------------------------------------
# Plain (non-pivoting) Householder QR of an m x n matrix A, m >= n.
# ------------------------------------------------------------------------------
def householder_qr(A: Mat, mut Q_out: Mat, mut R_out: Mat):
    var m = A.rows()
    var n = A.cols()

    var R = Mat(m, n)
    R.copyFrom(A)
    var Q = mat_identity(m, m)

    for k in range(n):
        var remaining = m - k
        if remaining <= 1:
            continue

        # Householder vector v for column k, rows k..m-1: v <- x - alpha*e1,
        # alpha = -sign(x0)*||x|| (sign chosen to avoid cancellation).
        var v = Vec(remaining)
        for i in range(remaining):
            v[i] = R[k + i, k]

        var xnorm = v.norm()
        if xnorm < REAL_MIN:
            continue

        var x0 = v[0]
        var alpha = -xnorm if x0 >= RealScalar(0) else xnorm
        v[0] = x0 - alpha

        var vnorm = v.norm()
        if vnorm < REAL_MIN:
            continue
        v.stableNormalize()

        # Apply H = I - 2vv^T to R[k:m, k:n] (zeroes column k below the
        # diagonal, updates the trailing columns).
        for j in range(k, n):
            var dot = RealScalar(0)
            for i in range(remaining):
                dot += v[i] * R[k + i, j]
            var factor = RealScalar(2) * dot
            for i in range(remaining):
                var updated = R[k + i, j] - factor * v[i]
                R[k + i, j] = updated

        # Accumulate Q <- Q * H on the right, restricted to columns k:m —
        # after all n steps, Q = H_0 * H_1 * ... * H_{n-1}, i.e. the full
        # orthogonal factor such that A == Q * R.
        for i in range(m):
            var dot2 = RealScalar(0)
            for t in range(remaining):
                dot2 += Q[i, k + t] * v[t]
            var factor2 = RealScalar(2) * dot2
            for t in range(remaining):
                var updated2 = Q[i, k + t] - factor2 * v[t]
                Q[i, k + t] = updated2

    var Rtop = Mat(n, n)
    for i in range(n):
        for j in range(n):
            Rtop[i, j] = R[i, j] if i <= j else RealScalar(0)

    Q_out = Q^
    R_out = Rtop^

# ------------------------------------------------------------------------------
# Non-blocked two-sided Jacobi sweep — the IsComplex == false branch of
# Eigen's `internal::jacobi_svd_nonblocking_sweep`. A(n x n) is diagonalized
# in place; U and V (both with n rows/cols in the range being rotated)
# accumulate the left/right rotations.
#
# Threshold is adaptive: `precision * maxDiagEntry`, with maxDiagEntry updated
# (and the threshold recomputed) after every single rotation, exactly as in
# Eigen — not a fixed threshold computed once per sweep.
#
# Returns True once a full sweep makes no rotation at all (converged).
# ------------------------------------------------------------------------------
def jacobi_svd_square_sweep(mut A: Mat, mut U: Mat, mut V: Mat, max_sweeps: Int) -> Bool:
    var n = A.rows()
    if n <= 1:
        return True

    var considerAsZero = REAL_MIN
    var precision = RealScalar(2) * REAL_EPSILON
    var maxDiagEntry = A.diagonal().cwiseAbsMax()

    for _sweep in range(max_sweeps):
        var any_rotated = False
        var threshold = max(considerAsZero, precision * maxDiagEntry)

        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) > threshold or abs(A[q, p]) > threshold:
                    any_rotated = True
                    var rots = jacobi_svd_2x2(A, p, q)
                    var j_left = rots[0]
                    var j_right = rots[1]
                    apply_rotation_left_rows(A, p, q, j_left)
                    apply_rotation_right_cols(A, p, q, j_right)
                    apply_rotation_cols_direct(U, p, q, j_left)
                    apply_rotation_right_cols(V, p, q, j_right)

                    maxDiagEntry = max(maxDiagEntry, abs(A[p, p]))
                    maxDiagEntry = max(maxDiagEntry, abs(A[q, q]))
                    threshold = max(considerAsZero, precision * maxDiagEntry)

        if not any_rotated:
            return True

    return False

def finish_jacobi_svd(
    mut work: Mat,
    mut U_out: Mat,
    mut V_out: Mat,
    diag_size: Int,
    maxCoeff: RealScalar,
    mut S_out: Vec,
) -> ComputationInfo:
    var converged = jacobi_svd_square_sweep(work, U_out, V_out, 100)

    # Fix signs: Jacobi doesn't guarantee a nonnegative diagonal, so flip
    # the corresponding U column wherever it's negative.
    var S = Vec(diag_size)
    for i in range(diag_size):
        var v = work[i, i]
        if v < RealScalar(0.0):
            S[i] = -v
            for r in range(U_out.rows()):
                U_out[r, i] = -U_out[r, i]
        else:
            S[i] = v

    # Sort descending; keep U/V columns in lockstep (selection sort — n is
    # expected small: this is always the base-case / already-square solver).
    for i in range(diag_size):
        var best = i
        for j in range(i + 1, diag_size):
            if S[j] > S[best]:
                best = j
        if best != i:
            swap(S[i], S[best])
            U_out.swap_cols(i, best)
            V_out.swap_cols(i, best)

    # Undo the initial max-coefficient scaling.
    for i in range(diag_size):
        S[i] = S[i] * maxCoeff

    S_out = S^

    return INFO_SUCCESS if converged else INFO_NO_CONVERGENCE

# ------------------------------------------------------------------------------
# Full pipeline, mirroring `JacobiSVD::compute_impl`:
#
#   1. Scale by the max abs coefficient (undone on the way out) to dodge
#      over/underflow.
#   2. R-SVD step: if non-square, QR-precondition down to a diagSize x
#      diagSize square problem (diagSize = min(rows, cols)); square inputs
#      skip straight to step 3.
#   3. Two-sided Jacobi sweep on the square work matrix.
#   4. Sign-fix the diagonal (flip the corresponding U column where negative).
#   5. Sort singular values descending, permuting U/V columns to match.
# ------------------------------------------------------------------------------
def jacobi_svd(A: Mat, mut U_out: Mat, mut S_out: Vec, mut V_out: Mat) -> ComputationInfo:
    var rows = A.rows()
    var cols = A.cols()
    var diag_size = rows if rows < cols else cols

    var maxCoeff = A.cwiseAbsMax()
    if maxCoeff == RealScalar(0):
        # Zero matrix: SVD is trivially all-zero singular values with
        # identity singular vectors.
        U_out = mat_identity(rows, rows)
        V_out = mat_identity(cols, cols)
        S_out = Vec(diag_size)
        return INFO_SUCCESS

    var scaled = Mat(rows, cols)
    for j in range(cols):
        for i in range(rows):
            scaled[i, j] = A[i, j] / maxCoeff

    if rows == cols:
        # Already square: no QR preconditioning needed.
        var work = Mat(rows, rows)
        work.copyFrom(scaled)
        U_out = mat_identity(rows, rows)
        V_out = mat_identity(cols, cols)
        return finish_jacobi_svd(work, U_out, V_out, diag_size, maxCoeff, S_out)
    elif rows > cols:
        # R-SVD, rows > cols case (Eigen's PreconditionIfMoreRowsThanCols):
        # A == Q * [R_top; 0], so diagonalizing R_top (cols x cols) gives
        # U_small, V such that A == (Q * [U_small; 0-embedded]) * S * V^T.
        # Since Q is already the full rows x rows orthogonal factor, "embed
        # U_small into Q" is exactly "let the sweep rotate the first `cols`
        # columns of Q in place" — no separate embedding step required.
        var Rtop = Mat(0, 0)
        householder_qr(scaled, U_out, Rtop)
        V_out = mat_identity(cols, cols)
        return finish_jacobi_svd(Rtop, U_out, V_out, diag_size, maxCoeff, S_out)
    else:
        # R-SVD, cols > rows case (Eigen's PreconditionIfMoreColsThanRows):
        # mirror image of the above, QR'ing A^T instead so V absorbs the
        # orthogonal factor and U comes out directly at its final size.
        var At = mat_transpose(scaled)
        var Rtop = Mat(0, 0)
        householder_qr(At, V_out, Rtop)
        var work = mat_transpose(Rtop)
        U_out = mat_identity(rows, rows)
        return finish_jacobi_svd(work, U_out, V_out, diag_size, maxCoeff, S_out)

struct JacobiSVD:
    var compute_v: Bool
    var u: Mat
    var v: Mat
    var sing_vals: Vec
    var status: ComputationInfo

    @always_inline
    def __init__(out self, compute_v: Bool):
        self.compute_v = compute_v
        self.u = Mat(0, 0)
        self.v = Mat(0, 0)
        self.sing_vals = Vec(0)
        self.status = INFO_SUCCESS

    def compute(mut self, m: Mat):
        # Computes both U and V internally regardless of
        # `self.compute_v` — the two-sided sweep needs the V-side rotations
        # to correctly evolve the work matrix and hence U/singular values
        # either way, so skipping V's accumulation wouldn't save much.
        self.status = jacobi_svd(m, self.u, self.sing_vals, self.v)

    @always_inline
    def info(self) -> ComputationInfo:
        return self.status

    @always_inline
    def matrixU(self) -> Mat:
        return self.u.block(0, 0, self.u.rows(), self.u.cols())

    @always_inline
    def matrixV(self) -> Mat:
        return self.v.block(0, 0, self.v.rows(), self.v.cols())

    @always_inline
    def singularValues(self) -> Vec:
        return self.sing_vals.segment(0, len(self.sing_vals))
