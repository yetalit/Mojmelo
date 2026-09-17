# ==============================================================================
# Based on libeigen — Eigen/src/SVD/UpperBidiagonalization.h, Copyright (C) 2009-2010 Benoit
# Jacob, Copyright (C) 2013-2014 Gael Guennebaud, MPL-2.0.
# ==============================================================================

from std.math import sqrt, hypot
from std.algorithm import vectorize
from mojmelo.utils.algorithm import parallelize
from .linalg_core import RealScalar, Vec, Mat, SIMD_WIDTH, matmul, mat_transpose, vec_dot

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

@always_inline
def _householder_right_update_row(
    row_ptr0: Pointer[RealScalar, MutUntrackedOrigin],
    col_stride: Int,
    ess_data: Pointer[RealScalar, MutUntrackedOrigin],
    ess_stride: Int,
    ess_len: Int,
    tau: RealScalar,
):
    # row_ptr0 -> M[i, 0]; M[i, j] lives at row_ptr0 + j*col_stride
    var s = row_ptr0[]
    for j in range(ess_len):
        s += ess_data[unsafe_offset=j * ess_stride] * row_ptr0[unsafe_offset=(j + 1) * col_stride]
    s = s * tau
    row_ptr0[] = row_ptr0[] - s
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

# ------------------------------------------------------------------------------
# Compact-WY blocked application of a panel of `plen` Householder reflectors.
# Net effect on C is the same as calling apply_householder_left once per
# reflector in decreasing index order (i.e. H applied first, then H
# one index down, etc.) — but batched into 2 GEMMs instead of `plen` rank-1
# updates, so the work rides the matmul kernel instead of doing memory-bound
# rank-1 updates one at a time.
#
# V: rows x plen. Column j is [zeros(j); 1; essential_j] — i.e. reflector j
# (0-indexed, increasing j = increasing original index) has its implicit
# leading 1 at row j and its essential tail below that. taus[j] pairs with
# column j. This is exactly LAPACK's DLARFT "forward" convention, giving
# H = H_0 * H_1 * ... * H_{plen-1} = I - V*T*V^T, which is the product our
# callers need (index 0 = outermost/leftmost = smallest original index,
# applied last; index plen-1 = innermost = largest original index, applied
# first — matching the existing high-to-low loop order).
# ------------------------------------------------------------------------------
@always_inline
def apply_compact_wy_block(mut C: Mat, V: Mat, taus: Vec, use_transpose: Bool = False):
    var rows = V.rows()
    var plen = V.cols()
    if plen == 0:
        return

    # Build T (plen x plen, upper triangular): T[0,0] = tau_0; for j > 0,
    # T[0:j,j] = -tau_j * T[0:j,0:j] * (V[:,0:j]^T v_j), T[j,j] = tau_j.
    # This stays a serial O(plen^2 * rows) loop (SIMD, not threaded).
    var T = Mat(plen, plen)
    T[0, 0] = taus[0]
    for j in range(1, plen):
        var vj = V.col(j)
        var z = Vec(j)
        for c in range(j):
            z[c] = vec_dot(V.col(c), vj)
        for r in range(j):
            var s = RealScalar(0)
            for c in range(r, j):
                s += T[r, c] * z[c]
            T[r, j] = -taus[j] * s
        T[j, j] = taus[j]

    # C <- C - V * (T * (V^T * C))   [Q = I - V T V^T applied], or with
    # T^T in place of T when the caller needs Q^T instead (same V, T —
    # (I - V T V^T)^T == I - V T^T V^T since V^T V^T-conjugation is its
    # own transpose-partner here). Q^T is what a QR/bidiagonalization
    # panel needs when flushing its effect onto trailing columns during
    # factorization; Q itself is what reconstructing U/V from an already-
    # finished factorization needs (apply_u_on_left etc.).
    var Vt = mat_transpose(V)
    var W = matmul(Vt, C)
    var TW: Mat
    if use_transpose:
        TW = matmul(mat_transpose(T), W)
    else:
        TW = matmul(T, W)
    var update = matmul(V, TW)
    for j in range(C.cols()):
        for i in range(rows):
            C[i, j] = C[i, j] - update[i, j]

# ------------------------------------------------------------------------------
# Small dense matrix-vector helpers used only by the blocked bidiagonalization
# below (the panel bookkeeping needs plain Mat*Vec / Mat^T*Vec, not the full
# Mat*Mat GEMM machinery).
# ------------------------------------------------------------------------------
@always_inline
def matTvec(A: Mat, x: Vec) -> Vec:
    """Y = A^T * X. y has length A.cols(), x must have length A.rows()."""
    var m = A.rows()
    var n = A.cols()
    var y = Vec(n)
    if m * max(n, 1) < 4096:
        for j in range(n):
            y[j] = vec_dot(A.col(j), x)
    else:
        @__parameter
        def process_col(j: Int):
            y[j] = vec_dot(A.col(j), x)
        parallelize[process_col](n)
    return y^

@always_inline
def matvec(A: Mat, x: Vec) -> Vec:
    """Y = A * X, via column-scaled accumulation (each column read once,
    contiguous). y has length A.rows(), x must have length A.cols()."""
    var m = A.rows()
    var n = A.cols()
    var y = Vec(m)
    for j in range(n):
        vec_add_scaled_inplace(y, A.col(j), x[j])
    return y^

@always_inline
def vec_sub_scaled_inplace(mut y: Vec, z: Vec, var scale: RealScalar):
    """Y -= scale * Z (elementwise, same length)."""
    var n = len(y)
    if y.stride == 1 and z.stride == 1:
        var yd = y.data
        var zd = z.data
        def sub[simd_width: Int](idx: Int) {mut}:
            yd.unsafe_store[simd_width](idx, yd.unsafe_load[simd_width](idx) - scale * zd.unsafe_load[simd_width](idx))
        vectorize[SIMD_WIDTH](n, sub)
    else:
        for i in range(n):
            y[i] = y[i] - scale * z[i]

@always_inline
def vec_add_scaled_inplace(mut y: Vec, z: Vec, scale: RealScalar):
    """Y += scale * Z."""
    vec_sub_scaled_inplace(y, z, -scale)

@always_inline
def vec_scale_inplace(mut y: Vec, var scale: RealScalar):
    var n = len(y)
    if y.stride == 1:
        var yd = y.data
        def sc[simd_width: Int](idx: Int) {mut}:
            yd.unsafe_store[simd_width](idx, yd.unsafe_load[simd_width](idx) * scale)
        vectorize[SIMD_WIDTH](n, sc)
    else:
        for i in range(n):
            y[i] = y[i] * scale

# ==============================================================================
# Port of Eigen's upperbidiagonalization_blocked_helper / upperbidiagonalization_inplace_blocked,
# implementing "The Design of a Parallel Dense Linear Algebra Software Library:
# Reduction to Hessenberg, Tridiagonal, and Bidiagonal Form" (Choi, Dongarra,
# Walker, 1995), section 3.3.
#
# Reduces the leading `bs` columns/rows of the panel A to bidiagonal form,
# while accumulating auxiliary matrices X, Y so the trailing A22 block can be
# updated with 2 GEMMs (A22 -= A10*Y_bottom^T + X_bottom*A01) instead of `bs`
# separate full-width rank-1 updates.
#
# Storage convention matches upperbidiagonalization_unblocked exactly: on
# return, A[k,k] holds tau for left reflector k (diagonal[k] holds beta),
# A[k,k+1] holds tau for right reflector k (upper_diagonal[k] holds beta),
# with essential vectors below/right of those pivots.
# ==============================================================================
def upperbidiagonalization_blocked_helper(mut A: Mat, mut diagonal: Vec, mut upper_diagonal: Vec, bs: Int, mut X: Mat, mut Y: Mat):
    var brows = A.rows()
    var bcols = A.cols()

    var tau_v: RealScalar
    var tau_u: RealScalar
    var tau_u_prev = RealScalar(0)

    for k in range(bs):
        var remainingRows = brows - k
        var remainingCols = bcols - k - 1

        var X_k1 = X.block(k, 0, remainingRows, k)
        var V_k1 = A.block(k, 0, remainingRows, k)

        # 1 - update the k-th column of A
        var v_k = A.col(k).tail(remainingRows)
        if k > 0:
            vec_sub_scaled_inplace(v_k, matvec(V_k1, Y.row(k).head(k)), RealScalar(1))
            vec_sub_scaled_inplace(v_k, matvec(X_k1, A.col(k).head(k)), RealScalar(1))

        # 2 - construct left Householder transform in-place
        var tb_v = make_householder_in_place(v_k)
        tau_v = tb_v[0]
        diagonal[k] = tb_v[1]

        if k + 1 < bcols:
            var Y_k = Y.block(k + 1, 0, remainingCols, k + 1)
            var U_k1 = A.block(0, k + 1, k, remainingCols)

            # this eases the application of Householder transforms below:
            # A(k,k) temporarily reads as the implicit "1" of v_k.
            A[k, k] = RealScalar(1)

            # 3 - y_k = tau_v * (A^T*v_k - Y_k[:,:k]*(V_k1^T*v_k) - U_k1^T*(X_k1^T*v_k))
            var y_k = Y.col(k).tail(remainingCols)
            y_k.copyFrom(matTvec(A.block(k, k + 1, remainingRows, remainingCols), v_k))
            vec_sub_scaled_inplace(y_k, matvec(Y_k.block(0, 0, remainingCols, k), matTvec(V_k1, v_k)), RealScalar(1))
            vec_sub_scaled_inplace(y_k, matTvec(U_k1, matTvec(X_k1, v_k)), RealScalar(1))
            vec_scale_inplace(y_k, tau_v)

            # 4 - update k-th row of A (it becomes u_k)
            var u_k = A.row(k).tail(remainingCols)
            vec_sub_scaled_inplace(u_k, matvec(Y_k, A.row(k).head(k + 1)), RealScalar(1))
            if k > 0:
                vec_sub_scaled_inplace(u_k, matTvec(U_k1, X.row(k).head(k)), RealScalar(1))

            # 5 - construct right Householder transform in-place
            var tb_u = make_householder_in_place(u_k)
            tau_u = tb_u[0]
            upper_diagonal[k] = tb_u[1]

            # A(k,k+1) temporarily reads as the implicit "1" of u_k.
            A[k, k + 1] = RealScalar(1)

            # 6 - x_k = tau_u * (A*u_k - X_k1[1:]*(U_k1*u_k) - A[k+1:,:k+1]*(Y_k^T*u_k))
            if remainingRows - 1 > 0:
                var x_k = X.col(k).segment(k + 1, remainingRows - 1)
                x_k.copyFrom(matvec(A.block(k + 1, k + 1, remainingRows - 1, remainingCols), u_k))
                vec_sub_scaled_inplace(x_k, matvec(X_k1.block(1, 0, remainingRows - 1, k), matvec(U_k1, u_k)), RealScalar(1))
                vec_sub_scaled_inplace(x_k, matvec(A.block(k + 1, 0, remainingRows - 1, k + 1), matTvec(Y_k, u_k)), RealScalar(1))
                vec_scale_inplace(x_k, tau_u)

            # Restore the PREVIOUS iteration's right-reflector pivot only now
            # -- row k-1 (via U_k1, which includes it) is still read as "1"
            # by this same iteration's step 4 above.
            if k > 0:
                A[k - 1, k] = tau_u_prev
            tau_u_prev = tau_u
        else:
            A[k - 1, k] = tau_u_prev

        A[k, k] = tau_v

    if bs < bcols:
        A[bs - 1, bs] = tau_u_prev

    # Flush the panel's accumulated effect onto A22 via 2 GEMMs.
    if bcols > bs and brows > bs:
        var A11 = A.block(bs, bs, brows - bs, bcols - bs)
        var A10 = A.block(bs, 0, brows - bs, bs)
        var A01 = A.block(0, bs, bs, bcols - bs)
        var Y_bottom = Y.block(bs, 0, bcols - bs, bs)
        var X_bottom = X.block(bs, 0, brows - bs, bs)

        # A01's row (bs-1) is the last right reflector's own row; its pivot
        # (A01[bs-1,0] == A[bs-1,bs]) must read as 1 for this GEMM, same
        # trick as within the loop -- temporarily override, then restore.
        var tmp = A[bs - 1, bs]
        A[bs - 1, bs] = RealScalar(1)

        var corr_a = matmul(A10, mat_transpose(Y_bottom))
        for j in range(A11.cols()):
            for i in range(A11.rows()):
                A11[i, j] = A11[i, j] - corr_a[i, j]

        var corr_b = matmul(X_bottom, A01)
        for j in range(A11.cols()):
            for i in range(A11.rows()):
                A11[i, j] = A11[i, j] - corr_b[i, j]

        A[bs - 1, bs] = tmp

def upperbidiagonalization_inplace_blocked(mut A: Mat, mut diag: Vec, mut superdiag: Vec, max_block_size: Int = 16):
    var rows = A.rows()
    var cols = A.cols()
    var size = min(rows, cols)
    if size == 0:
        return
    var X = Mat(rows, max_block_size)
    var Y = Mat(cols, max_block_size)
    var block_size = min(max_block_size, size)

    var k = 0
    while k < size:
        var bs = min(size - k, block_size)
        var brows = rows - k
        var bcols = cols - k
        var B = A.block(k, k, brows, bcols)
        var diag_sub = diag.segment(k, len(diag) - k)
        var superdiag_sub = superdiag.segment(k, max(len(superdiag) - k, 0))

        if k + bs == cols or bcols < 2 * block_size:
            # Fall back to unblocked for the small trailing submatrix.
            upperbidiagonalization_unblocked(B, diag_sub, superdiag_sub)
            break
        else:
            var X_sub = X.block(0, 0, brows, bs)
            var Y_sub = Y.block(0, 0, bcols, bs)
            upperbidiagonalization_blocked_helper(B, diag_sub, superdiag_sub, bs, X_sub, Y_sub)

        k += bs

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
        upperbidiagonalization_inplace_blocked(self.m_householder, self.m_diag, self.m_superdiag)
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
        comptime block_size = 64
        var k_hi = self.m_cols - 1
        while k_hi >= 0:
            var plen = min(block_size, k_hi + 1)
            var kb = k_hi - plen + 1
            var block_rows = self.m_rows - kb
            var V = Mat(block_rows, plen)
            var taus = Vec(plen)
            for j in range(plen):
                var k = kb + j
                taus[j] = self.m_householder[k, k]
                V[j, j] = RealScalar(1)
                var essential = self.m_householder.col(k).segment(k + 1, self.m_rows - k - 1)
                V.col(j).segment(j + 1, len(essential)).copyFrom(essential)
            var C = M.block(kb, 0, block_rows, M.cols())
            apply_compact_wy_block(C, V, taus)
            k_hi = kb - 1

    # Apply V_h = H_0 * H_1 * ... * H_{cols-2} to M from the left, i.e.
    # M <- V_h * M. Same reverse application order as apply_u_on_left; the
    # V-side reflectors are shifted one column/row relative to U's, since
    # the (0,0) entry of a bidiagonal's right-hand transform is untouched.
    @always_inline
    def apply_v_on_left(self, mut M: Mat):
        comptime block_size = 64
        var k_hi = self.m_cols - 2
        while k_hi >= 0:
            var plen = min(block_size, k_hi + 1)
            var kb = k_hi - plen + 1
            var pivot0 = kb + 1  # row of reflector j=0's implicit leading 1
            var block_rows = self.m_cols - pivot0
            var V = Mat(block_rows, plen)
            var taus = Vec(plen)
            for j in range(plen):
                var k = kb + j
                taus[j] = self.m_householder[k, k + 1]
                V[j, j] = RealScalar(1)
                var essential = self.m_householder.row(k).segment(k + 2, self.m_cols - k - 2)
                V.col(j).segment(j + 1, len(essential)).copyFrom(essential)
            var C = M.block(pivot0, 0, block_rows, M.cols())
            apply_compact_wy_block(C, V, taus)
            k_hi = kb - 1
