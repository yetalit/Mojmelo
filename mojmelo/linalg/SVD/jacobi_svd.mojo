# ==============================================================================
# Based on libeigen — Eigen/src/SVD/JacobiSVD.h, Copyright (C) 2009-2010 Benoit
# Jacob, Copyright (C) 2013-2014 Gael Guennebaud, MPL-2.0.
# ------------------------------------------------------------------------------
# Eigen's JacobiSVD reduces a rectangular m x p matrix to a square diagSize x
# diagSize problem (diagSize = min(m, p)) via a QR decomposition first (the
# "R-SVD" step), then runs the two-sided Jacobi sweep on that square work
# matrix.
#
# CONVERGENCE: capped at `max_sweeps` cyclic sweeps over all (p,q) pairs;
# returns INFO_NO_CONVERGENCE (not a crash) if that cap is hit without every
# off-diagonal pair falling under the threshold — mirrors Eigen's own
# NoConvergence outcome (Eigen itself has no such cap and simply loops until
# converged; the cap here is a deliberate safety valve).
# ==============================================================================

from std.math import sqrt
from .linalg_core import (
    RealScalar,
    REAL_EPSILON,
    REAL_MIN,
    Vec,
    IVec,
    Mat,
    mat_identity,
    mat_transpose,
)
from .jacobi import (
    jacobi_svd_2x2,
    apply_rotation_left_rows,
    apply_rotation_right_cols,
    apply_rotation_cols_direct,
)
from .bidiagonalization import (
    make_householder_in_place,
    apply_householder_left,
)
from .bdcsvd import (
    ComputationInfo,
    INFO_SUCCESS,
    INFO_NO_CONVERGENCE,
)

# ------------------------------------------------------------------------------
# Column-pivoted Householder QR of an m x n matrix A, m >= n: A*P = Q*R for a
# column permutation P. At each of the n steps, the remaining column with the
# largest norm is swapped into the pivot position before the Householder
# reflection is built.
#
# Column norms are maintained incrementally rather than recomputed from
# scratch every step (an O(n) update instead of O(m)), using the same
# stable downdate LAPACK's xGEQP3 uses (Drmac & Bujanovic, LAWN176): the
# updated norm is trusted only while updated/direct stays above a threshold;
# once it decays past that, the norm for that column is recomputed exactly.
# Skipping that fallback is a known way for column-pivoted QR to silently
# pick bad pivots on exactly the ill-conditioned inputs this exists to help.
# ------------------------------------------------------------------------------
struct ColPivHouseholderQR:
    var m_qr: Mat
    var m_hCoeffs: Vec
    # m_colsPermutation[k] = index of the original column now sitting in
    # position k of m_qr (i.e. m_qr[:, k] == A[:, m_colsPermutation[k]]).
    var m_colsPermutation: IVec
    var m_rows: Int
    var m_cols: Int  # number of reflectors == min(original rows, cols)
    var m_maxPivot: RealScalar

    @always_inline
    def __init__(out self):
        self.m_qr = Mat(0, 0)
        self.m_hCoeffs = Vec(0)
        self.m_colsPermutation = IVec(0)
        self.m_rows = 0
        self.m_cols = 0
        self.m_maxPivot = RealScalar(0)

    def compute(mut self, A: Mat):
        var rows = A.rows()
        var cols = A.cols()
        self.m_qr = A.copy()
        self.m_rows = rows
        var size = min(rows, cols)
        self.m_cols = size
        self.m_hCoeffs = Vec(size)
        self.m_maxPivot = RealScalar(0)

        self.m_colsPermutation = IVec(cols)
        for j in range(cols):
            self.m_colsPermutation[j] = j
        if size == 0:
            return

        # colNormsUpdated: cheaply-downdated running estimate of each
        # remaining column's norm. colNormsDirect: the norm as of its last
        # exact computation — comparing the two is how the downdate's
        # accuracy is monitored.
        var colNormsUpdated = Vec(cols)
        var colNormsDirect = Vec(cols)
        for j in range(cols):
            var nrm = self.m_qr.col(j).norm()
            colNormsUpdated[j] = nrm
            colNormsDirect[j] = nrm

        var norm_downdate_threshold = sqrt(REAL_EPSILON)

        for k in range(size):
            # Pivot: bring the remaining column with the largest (updated)
            # norm into position k.
            var biggest = k
            var biggest_norm = colNormsUpdated[k]
            for j in range(k + 1, cols):
                if colNormsUpdated[j] > biggest_norm:
                    biggest_norm = colNormsUpdated[j]
                    biggest = j

            if biggest != k:
                self.m_qr.swap_cols(k, biggest)
                swap(colNormsUpdated[k], colNormsUpdated[biggest])
                swap(colNormsDirect[k], colNormsDirect[biggest])
                swap(self.m_colsPermutation[k], self.m_colsPermutation[biggest])

            var remainingRows = rows - k
            var remainingCols = cols - k - 1

            var col_tail = self.m_qr.col(k).segment(k, remainingRows)
            var tb = make_householder_in_place(col_tail)
            var tau = tb[0]
            var beta = tb[1]
            self.m_hCoeffs[k] = tau
            # make_householder_in_place aliased tau into col_tail[0] ==
            # m_qr[k,k]; put the true R diagonal value back.
            self.m_qr[k, k] = beta

            var absBeta = abs(beta)
            if absBeta > self.m_maxPivot:
                self.m_maxPivot = absBeta

            if remainingCols > 0 and tau != RealScalar(0):
                var essential = col_tail.segment(1, remainingRows - 1)
                var sub = self.m_qr.block(k, k + 1, remainingRows, remainingCols)
                apply_householder_left(sub, essential, tau)

            # LAWN176 column-norm downdate (same derivation LAPACK's
            # xGEQP3/xGEQPF use): after reflecting, each remaining column's
            # new tail norm is exactly the old norm scaled by
            # sqrt(1 - (element just zeroed / old norm)^2), computed here in
            # the numerically nicer (1+t)(1-t) form. If that estimate's
            # relative trust (tracked via colNormsUpdated/colNormsDirect)
            # has decayed too far, fall back to an exact recomputation
            # rather than keep compounding an unreliable update.
            for j in range(k + 1, cols):
                if colNormsUpdated[j] != RealScalar(0):
                    var temp = abs(self.m_qr[k, j]) / colNormsUpdated[j]
                    temp = max(RealScalar(0), (RealScalar(1) + temp) * (RealScalar(1) - temp))
                    var ratio = colNormsUpdated[j] / colNormsDirect[j]
                    var temp2 = temp * ratio * ratio
                    if temp2 <= norm_downdate_threshold:
                        var recomputed = self.m_qr.col(j).segment(k + 1, rows - k - 1).norm()
                        colNormsDirect[j] = recomputed
                        colNormsUpdated[j] = recomputed
                    else:
                        colNormsUpdated[j] = colNormsUpdated[j] * sqrt(temp)

    @always_inline
    def matrixR(self) -> Mat:
        """The cols x cols (== m_cols x m_cols) upper-triangular R factor of
        A*P, as a fresh dense copy.
        """
        var q = self.m_cols
        var R = Mat(q, q)
        for j in range(q):
            for i in range(j + 1):
                R[i, j] = self.m_qr[i, j]
        return R^

    @always_inline
    def apply_q_on_left(self, mut M: Mat):
        """M <- Q * M, i.e. H_0 * H_1 * ... * H_{m_cols-1} * M — reflectors
        applied in reverse order, same pattern as
        UpperBidiagonalization.apply_u_on_left / bdcsvd.HouseholderQR.
        """
        var k = self.m_cols - 1
        while k >= 0:
            var tau = self.m_hCoeffs[k]
            if tau != RealScalar(0):
                var essential = self.m_qr.col(k).segment(k + 1, self.m_rows - k - 1)
                var sub = M.block(k, 0, self.m_rows - k, M.cols())
                apply_householder_left(sub, essential, tau)
            k -= 1

# ------------------------------------------------------------------------------
# Undo a ColPivHouseholderQR's column permutation on the singular-vector
# matrix that came out of the square Jacobi sweep on R: since A*P = Q*R,
# the sweep produces the singular vectors of the *permuted* problem, so row k
# of that result belongs at row perm[k] (the original, unpermuted index).
# ------------------------------------------------------------------------------
@always_inline
def unpermute_rows(mut M: Mat, perm: IVec):
    var n = len(perm)
    var src = M.copy()
    for k in range(n):
        M.row(perm[k]).copyFrom(src.row(k))

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
        if v < RealScalar(0):
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
def jacobi_svd(
    A: Mat, mut U_out: Mat, mut S_out: Vec, mut V_out: Mat, thinU: Bool = False, thinV: Bool = False
) -> ComputationInfo:
    var rows = A.rows()
    var cols = A.cols()
    var diag_size = rows if rows < cols else cols

    var Ucols = diag_size if thinU else rows
    var Vcols = diag_size if thinV else cols

    var maxCoeff = A.cwiseAbsMax()
    if maxCoeff == RealScalar(0):
        # Zero matrix: SVD is trivially all-zero singular values with
        # identity singular vectors.
        U_out = mat_identity(rows, Ucols)
        V_out = mat_identity(cols, Vcols)
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
        U_out = mat_identity(rows, Ucols)
        V_out = mat_identity(cols, Vcols)
        return finish_jacobi_svd(work, U_out, V_out, diag_size, maxCoeff, S_out)
    elif rows > cols:
        # R-SVD, rows > cols case (Eigen's PreconditionIfMoreRowsThanCols):
        # A*P == Q * [R_top; 0] for the column permutation P that
        # ColPivHouseholderQR chose, so diagonalizing R_top (cols x cols)
        # gives U_small, V_perm such that A*P == (Q * [U_small;
        # 0-embedded]) * S * V_perm^T. Q is already the full rows x rows
        # orthogonal factor (built via apply_q_on_left on the identity), so
        # "embed U_small into Q" is exactly "let the sweep rotate the first
        # `cols` columns of Q in place" — no separate embedding step
        # required. V_perm's *rows* (indexed by A's original columns) still
        # need un-permuting to undo P before they're the true V.
        #
        # Thin U: build only an rows x diag_size (== rows x cols) slice of
        # the identity before applying Q. apply_q_on_left's cost scales
        # with the target's column count, so this is the whole saving —
        # the sweep afterwards only ever touches columns < diag_size
        # anyway, full or thin. V is already diag_size wide here regardless
        # of thinV, since cols == diag_size in this branch.
        var qr = ColPivHouseholderQR()
        qr.compute(scaled)
        var Rtop = qr.matrixR()
        U_out = mat_identity(rows, Ucols)
        qr.apply_q_on_left(U_out)
        V_out = mat_identity(cols, Vcols)
        var info = finish_jacobi_svd(Rtop, U_out, V_out, diag_size, maxCoeff, S_out)
        unpermute_rows(V_out, qr.m_colsPermutation)
        return info
    else:
        # R-SVD, cols > rows case (Eigen's PreconditionIfMoreColsThanRows):
        # mirror image of the above, column-pivoted-QR'ing A^T instead so V
        # absorbs the orthogonal factor and U comes out directly at its
        # final size. The permutation this time reorders A's *rows* (== A^T's
        # columns), so it's U's rows that need un-permuting afterwards.
        # Mirror image of the thin note above: here it's V that holds Q, so
        # thinV is what narrows the expensive apply_q_on_left; U is already
        # diag_size wide regardless of thinU, since rows == diag_size here.
        var At = mat_transpose(scaled)
        var qr = ColPivHouseholderQR()
        qr.compute(At)
        var Rtop = qr.matrixR()
        V_out = mat_identity(cols, Vcols)
        qr.apply_q_on_left(V_out)
        var work = mat_transpose(Rtop)
        U_out = mat_identity(rows, Ucols)
        var info = finish_jacobi_svd(work, U_out, V_out, diag_size, maxCoeff, S_out)
        unpermute_rows(U_out, qr.m_colsPermutation)
        return info

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

    def compute(mut self, m: Mat, thinU: Bool = False, thinV: Bool = False):
        # Computes both U and V internally regardless of
        # `self.compute_v` — the two-sided sweep needs the V-side rotations
        # to correctly evolve the work matrix and hence U/singular values
        # either way, so skipping V's accumulation wouldn't save much.
        self.status = jacobi_svd(m, self.u, self.sing_vals, self.v, thinU, thinV)

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
