# ==============================================================================
# Based on libeigen — Eigen/src/SVD/UpperBidiagonalization.h, Copyright (C) 2009-2010 Benoit
# Jacob, Copyright (C) 2013-2014 Gael Guennebaud, MPL-2.0.
# ------------------------------------------------------------------------------
# This ports `upperbidiagonalization_inplace_unblocked` — the plain O(n^3),
# no-fancy-optimizations version, which the original says is "faster for
# small matrix sizes". `upperbidiagonalization_inplace_blocked` (the
# WY-representation, GEMM-heavy version for large matrices) is NOT ported.
# ==============================================================================

from std.math import sqrt, hypot
from std.algorithm import vectorize
from mojmelo.utils.algorithm import parallelize
from .linalg_core import RealScalar, Vec, Mat, SIMD_WIDTH, matmul, mat_transpose

@always_inline
def make_householder_in_place(mut v: Vec) -> Tuple[RealScalar, RealScalar]:
    var m = len(v)
    if m == 1:
        var beta = v[0]
        v[0] = RealScalar(0)
        return (RealScalar(0), beta)

    var tail = v.segment(1, m - 1)
    var xnorm = tail.norm()
    var alpha = v[0]

    if xnorm == RealScalar(0):
        v[0] = RealScalar(0)
        return (RealScalar(0), alpha)

    var beta: RealScalar
    if alpha >= RealScalar(0):
        beta = -hypot(alpha, xnorm)
    else:
        beta = hypot(alpha, xnorm)

    var tau = (beta - alpha) / beta
    var scale = RealScalar(1) / (alpha - beta)
    for i in range(len(tail)):
        tail[i] = tail[i] * scale

    v[0] = tau
    return (tau, beta)

# ------------------------------------------------------------------------------
# Rank-1 Householder updates. Both are O(rows*cols)
# ------------------------------------------------------------------------------
@always_inline
def _householder_left_update_col(
    col_ptr: Pointer[RealScalar, MutUntrackedOrigin],
    var ess_data: Pointer[RealScalar, MutUntrackedOrigin],
    ess_stride: Int,
    ess_len: Int,
    tau: RealScalar,
):
    # col_ptr -> M[0, j]; M[1:, j] is contiguous from col_ptr+1 (row_stride
    # is always 1 in this codebase), regardless of j — that's what makes
    # parallelizing over columns safe and simple: each column is an
    # independent, contiguous rank-1 update.
    var tail_ptr = col_ptr.unsafe_offset(1)
    var s = col_ptr[]

    if ess_stride == 1:
        def dotv[simd_width: Int](idx: Int) {mut}:
            s += (ess_data.unsafe_load[width=simd_width](idx) * tail_ptr.unsafe_load[width=simd_width](idx)).reduce_add()
        vectorize[SIMD_WIDTH](ess_len, dotv)
    else:
        # Strided essential (V-side reflector via apply_v_on_left): plain
        # scalar loop rather than a SIMD path.
        for i in range(ess_len):
            s += ess_data[unsafe_offset=i * ess_stride] * tail_ptr[unsafe_offset=i]

    s = s * tau
    col_ptr[] = col_ptr[] - s

    if ess_stride == 1:
        def axpy[simd_width: Int](idx: Int) {mut}:
            var upd = tail_ptr.unsafe_load[width=simd_width](idx) - s * ess_data.unsafe_load[width=simd_width](idx)
            tail_ptr.unsafe_store[simd_width](idx, upd)
        vectorize[SIMD_WIDTH](ess_len, axpy)
    else:
        for i in range(ess_len):
            var upd = tail_ptr[unsafe_offset=i] - s * ess_data[unsafe_offset=i * ess_stride]
            tail_ptr[unsafe_offset=i] = upd

@always_inline
def apply_householder_left(mut M: Mat, essential: Vec, tau: RealScalar):
    """M <- (I - tau * w w^T) * M, where w = [1, essential...] has M.rows() entries."""
    if tau == RealScalar(0):
        return
    var ess_len = len(essential)
    var ess_stride = essential.stride
    var ess_data = essential.data
    var col_stride = M.col_stride
    var base = M.data
    var cols = M.cols()

    # Threshold avoids parallel-dispatch overhead dominating on small
    # blocks (e.g. deep in the recursion / small trailing columns).
    if cols * max(ess_len, 1) < 4096:
        for j in range(cols):
            _householder_left_update_col(base.unsafe_offset(j * col_stride), ess_data, ess_stride, ess_len, tau)
    else:
        @__parameter
        def process_col(j: Int):
            _householder_left_update_col(base.unsafe_offset(j * col_stride), ess_data, ess_stride, ess_len, tau)
        parallelize[process_col](cols)

# ------------------------------------------------------------------------------
# qr's upper triangle (including diagonal) holds R, qr's strict lower triangle
# holds each reflector's essential vector, and hCoeffs holds the tau's.
# The two structs differ only in how compute() picks/orders columns before
# reflecting — R extraction and applying Q are identical either way, so both
# structs delegate to these instead of keeping their own copies in sync by hand.
# ------------------------------------------------------------------------------
@always_inline
def householder_qr_matrixR(qr: Mat, size: Int) -> Mat:
    """The size x size upper-triangular R factor, as a fresh dense copy."""
    var R = Mat(size, size)
    for j in range(size):
        for i in range(j + 1):
            R[i, j] = qr[i, j]
    return R^

@always_inline
def householder_qr_apply_q_on_left(qr: Mat, hCoeffs: Vec, rows: Int, num_reflectors: Int, mut M: Mat):
    """M <- Q * M, i.e. H_0 * H_1 * ... * H_{num_reflectors-1} * M —
    reflectors applied in reverse order, same pattern as
    UpperBidiagonalization.apply_u_on_left.
    """
    var k = num_reflectors - 1
    while k >= 0:
        var tau = hCoeffs[k]
        if tau != RealScalar(0):
            var essential = qr.col(k).segment(k + 1, rows - k - 1)
            var sub = M.block(k, 0, rows - k, M.cols())
            apply_householder_left(sub, essential, tau)
        k -= 1

@always_inline
def _householder_right_update_row(
    row_ptr0: Pointer[RealScalar, MutUntrackedOrigin],
    var col_stride: Int,
    ess_data: Pointer[RealScalar, MutUntrackedOrigin],
    var ess_stride: Int,
    ess_len: Int,
    tau: RealScalar,
):
    # row_ptr0 -> M[i, 0]; M[i, j] lives at row_ptr0 + j*col_stride
    var s = row_ptr0[]

    if ess_stride == col_stride:
        var e_ptr = ess_data
        var m_ptr = row_ptr0.unsafe_offset(col_stride)
        def dotv[simd_width: Int](idx: Int) {mut}:
            var ev = e_ptr.unsafe_strided_load[width=simd_width](ess_stride)
            var mv = m_ptr.unsafe_strided_load[width=simd_width](col_stride)
            s += (ev * mv).reduce_add()
            e_ptr = e_ptr.unsafe_offset(simd_width * ess_stride)
            m_ptr = m_ptr.unsafe_offset(simd_width * col_stride)
        vectorize[SIMD_WIDTH](ess_len, dotv)
    else:
        for j in range(ess_len):
            s += ess_data[unsafe_offset=j * ess_stride] * row_ptr0[unsafe_offset=(j + 1) * col_stride]

    s = s * tau
    row_ptr0[] = row_ptr0[] - s

    if ess_stride == col_stride:
        var e_ptr2 = ess_data
        var m_ptr2 = row_ptr0.unsafe_offset(col_stride)
        def axpy[simd_width: Int](idx: Int) {mut}:
            var ev = e_ptr2.unsafe_strided_load[width=simd_width](ess_stride)
            var mv = m_ptr2.unsafe_strided_load[width=simd_width](col_stride)
            m_ptr2.unsafe_strided_store[width=simd_width](mv - s * ev, col_stride)
            e_ptr2 = e_ptr2.unsafe_offset(simd_width * ess_stride)
            m_ptr2 = m_ptr2.unsafe_offset(simd_width * col_stride)
        vectorize[SIMD_WIDTH](ess_len, axpy)
    else:
        for j in range(ess_len):
            var off = (j + 1) * col_stride
            row_ptr0[unsafe_offset=off] = row_ptr0[unsafe_offset=off] - s * ess_data[unsafe_offset=j * ess_stride]

@always_inline
def apply_householder_right(mut M: Mat, essential: Vec, tau: RealScalar):
    """M <- M * (I - tau * w w^T), where w = [1, essential...] has M.cols() entries."""
    if tau == RealScalar(0):
        return
    var ess_len = len(essential)
    var ess_stride = essential.stride
    var ess_data = essential.data
    var col_stride = M.col_stride
    var base = M.data
    var rows = M.rows()

    if rows * max(ess_len, 1) < 4096:
        for i in range(rows):
            _householder_right_update_row(base.unsafe_offset(i), col_stride, ess_data, ess_stride, ess_len, tau)
    else:
        @__parameter
        def process_row(i: Int):
            _householder_right_update_row(base.unsafe_offset(i), col_stride, ess_data, ess_stride, ess_len, tau)
        parallelize[process_row](rows)

def upperbidiagonalization_unblocked(mut mat: Mat, mut diag: Vec, mut superdiag: Vec):
    var rows = mat.rows()
    var cols = mat.cols()
    var k = 0
    while True:
        var remainingRows = rows - k
        var remainingCols = cols - k - 1

        # Left Householder: zero out column k below the diagonal.
        var col_tail = mat.col(k).segment(k, remainingRows)
        var tau_beta_v = make_householder_in_place(col_tail)
        var tau_v = tau_beta_v[0]
        diag[k] = tau_beta_v[1]
        if remainingCols > 0 and tau_v != RealScalar(0):
            var essential_v = col_tail.segment(1, remainingRows - 1)
            var sub = mat.block(k, k + 1, remainingRows, remainingCols)
            apply_householder_left(sub, essential_v, tau_v)

        if k == cols - 1:
            break

        # Right Householder: zero out row k to the right of the superdiagonal.
        var row_tail = mat.row(k).segment(k + 1, remainingCols)
        var tau_beta_u = make_householder_in_place(row_tail)
        var tau_u = tau_beta_u[0]
        superdiag[k] = tau_beta_u[1]
        if remainingRows - 1 > 0 and tau_u != RealScalar(0):
            var essential_u = row_tail.segment(1, remainingCols - 1)
            var sub2 = mat.block(k + 1, k + 1, remainingRows - 1, remainingCols)
            apply_householder_right(sub2, essential_u, tau_u)

        k += 1

struct UpperBidiagonalization:
    var m_householder: Mat
    var m_diag: Vec
    var m_superdiag: Vec
    var m_rows: Int
    var m_cols: Int
    var m_isInitialized: Bool

    @always_inline
    def __init__(out self):
        self.m_householder = Mat(0, 0)
        self.m_diag = Vec(0)
        self.m_superdiag = Vec(0)
        self.m_rows = 0
        self.m_cols = 0
        self.m_isInitialized = False

    def compute(mut self, A: Mat):
        # Precondition (matches Eigen): rows >= cols. bdcsvd.mojo's compute()
        # already transposes its input before handing it here, so this
        # should always hold by construction.
        var rows = A.rows()
        var cols = A.cols()
        self.m_rows = rows
        self.m_cols = cols
        self.m_householder = A.copy()
        self.m_diag = Vec(cols)
        self.m_superdiag = Vec(max(cols - 1, 0))
        upperbidiagonalization_unblocked(self.m_householder, self.m_diag, self.m_superdiag)
        self.m_isInitialized = True

    def compute_unblocked(mut self, A: Mat):
        # Parity with Eigen's computeUnblocked(): always the plain O(n^3)
        # kernel, regardless of matrix size.
        var rows = A.rows()
        var cols = A.cols()
        self.m_rows = rows
        self.m_cols = cols
        self.m_householder = A.copy()
        self.m_diag = Vec(cols)
        self.m_superdiag = Vec(max(cols - 1, 0))
        upperbidiagonalization_unblocked(self.m_householder, self.m_diag, self.m_superdiag)
        self.m_isInitialized = True

    @always_inline
    def bidiagonal_diagonal(self) -> Vec:
        return self.m_diag.segment(0, len(self.m_diag))

    @always_inline
    def bidiagonal_superdiagonal(self) -> Vec:
        return self.m_superdiag.segment(0, len(self.m_superdiag))

    # Apply U_h = H_0 * H_1 * ... * H_{cols-1} to M from the left, i.e.
    # M <- U_h * M. Since U_h*M = H_0*(H_1*(...*(H_{cols-1}*M))), apply the
    # highest-indexed reflector first and work back down to H_0.
    @always_inline
    def apply_u_on_left(self, mut M: Mat):
        var k = self.m_cols - 1
        while k >= 0:
            var tau = self.m_householder[k, k]
            if tau != RealScalar(0):
                var essential = self.m_householder.col(k).segment(k + 1, self.m_rows - k - 1)
                var sub = M.block(k, 0, self.m_rows - k, M.cols())
                apply_householder_left(sub, essential, tau)
            k -= 1

    # Apply V_h = H_0 * H_1 * ... * H_{cols-2} to M from the left, i.e.
    # M <- V_h * M. Same reverse application order as apply_u_on_left; the
    # V-side reflectors are shifted one column/row relative to U's, since
    # the (0,0) entry of a bidiagonal's right-hand transform is untouched.
    @always_inline
    def apply_v_on_left(self, mut M: Mat):
        var k = self.m_cols - 2
        while k >= 0:
            var tau = self.m_householder[k, k + 1]
            if tau != RealScalar(0):
                var essential = self.m_householder.row(k).segment(k + 2, self.m_cols - k - 2)
                var sub = M.block(k + 1, 0, self.m_cols - k - 1, M.cols())
                apply_householder_left(sub, essential, tau)
            k -= 1
