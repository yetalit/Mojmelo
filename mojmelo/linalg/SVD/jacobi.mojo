# Based on libeigen — Eigen/src/Jacobi/Jacobi.h, Copyright (C) 2009-2010 Benoit Jacob, Copyright (C) 2013-2014 Gael Guennebaud, MPL-2.0.

from std.math import sqrt
from .linalg_core import RealScalar, REAL_MIN, SIMD_WIDTH, Mat
from std.algorithm import vectorize

# ------------------------------------------------------------------------------
# A 2x2 Givens/Jacobi rotation, and its application to a pair of matrix
# columns over a row range.
# ------------------------------------------------------------------------------
@fieldwise_init
struct JacobiRotation(TrivialRegisterPassable):
    var c: RealScalar
    var s: RealScalar

def apply_jacobi_on_right(
    mut m: Mat, row_start: Int, row_count: Int, col_p: Int, col_q: Int, rot: JacobiRotation
):
    # Both m[:, col_p] and m[:, col_q] are contiguous (row_stride is always
    # 1 here), so this is a safe two-vector blend to vectorize — unlike the
    # Householder appliers, there's no strided operand to worry about.
    var p_ptr = m.data.unsafe_offset(row_start + col_p * m.col_stride)
    var q_ptr = m.data.unsafe_offset(row_start + col_q * m.col_stride)
    var c = rot.c
    var s = rot.s

    if row_count < 512:
        for r in range(row_count):
            var vp = p_ptr[unsafe_offset=r]
            var vq = q_ptr[unsafe_offset=r]
            p_ptr[unsafe_offset=r] = c * vp - s * vq
            q_ptr[unsafe_offset=r] = s * vp + c * vq
    else:
        def blend[simd_width: Int](idx: Int) {mut}:
            var vp = p_ptr.unsafe_load[width=simd_width](idx)
            var vq = q_ptr.unsafe_load[width=simd_width](idx)
            p_ptr.unsafe_store[simd_width](idx, c * vp - s * vq)
            q_ptr.unsafe_store[simd_width](idx, s * vp + c * vq)
        vectorize[SIMD_WIDTH](row_count, blend)

@always_inline
def apply_rotation_left_rows(mut M: Mat, p: Int, q: Int, rot: JacobiRotation):
    """M.rows[{p,q}] <- rot * M.rows[{p,q}]  (Eigen's applyOnTheLeft)."""
    for j in range(M.cols()):
        var xp = M[p, j]
        var xq = M[q, j]
        M[p, j] = rot.c * xp + rot.s * xq
        M[q, j] = -rot.s * xp + rot.c * xq

@always_inline
def apply_rotation_right_cols(mut M: Mat, p: Int, q: Int, rot: JacobiRotation):
    """M.cols[{p,q}] <- M.cols[{p,q}] * rot  (Eigen's applyOnTheRight)."""
    for i in range(M.rows()):
        var xp = M[i, p]
        var xq = M[i, q]
        M[i, p] = rot.c * xp - rot.s * xq
        M[i, q] = rot.s * xp + rot.c * xq

@always_inline
def apply_rotation_cols_direct(mut M: Mat, p: Int, q: Int, rot: JacobiRotation):
    for i in range(M.rows()):
        var xp = M[i, p]
        var xq = M[i, q]
        M[i, p] = rot.c * xp + rot.s * xq
        M[i, q] = -rot.s * xp + rot.c * xq

# ------------------------------------------------------------------------------
# finds (c, s) such that applying the resulting rotation on
# both sides of the symmetric 2x2 matrix [[x, y], [y, z]]
# diagonalizes it.
# ------------------------------------------------------------------------------
@always_inline
def make_jacobi(x: RealScalar, y: RealScalar, z: RealScalar) -> JacobiRotation:
    var abs_y = abs(y)
    var deno = RealScalar(2) * abs_y
    if deno < REAL_MIN:
        return JacobiRotation(RealScalar(1), RealScalar(0))

    var delta = x - z
    var abs_delta = abs(delta)

    # |t| = 1 / (|tau| + sqrt(1 + tau^2)), tau = delta / deno. Scale the
    # numerator/denominator pair by min(1, 1/|tau|) (scale = 1 at tau = 0) so
    # that only a ratio <= 1 ever gets squared, then normalize the pair
    # directly instead of forming |t| via a division.
    var numerator: RealScalar
    var denominator: RealScalar
    if abs_delta > deno:
        var ratio = deno / abs_delta
        numerator = ratio
        denominator = sqrt(RealScalar(1) + ratio * ratio) + RealScalar(1)
    else:
        var ratio = abs_delta / deno
        numerator = RealScalar(1)
        denominator = sqrt(RealScalar(1) + ratio * ratio) + ratio

    var n = RealScalar(1) / sqrt(numerator * numerator + denominator * denominator)
    var sine = numerator * n
    var sign_t = RealScalar(1) if delta > RealScalar(0) else RealScalar(-1)
    var signed_sine = -sign_t * sine
    # For real y, conj(y)/abs(y) is just its sign.
    var s = -signed_sine if y < RealScalar(0) else signed_sine
    var c = denominator * n
    return JacobiRotation(c, s)

@always_inline
def jacobi_svd_2x2(M: Mat, p: Int, q: Int) -> Tuple[JacobiRotation, JacobiRotation]:
    var m00 = M[p, p]
    var m01 = M[p, q]
    var m10 = M[q, p]
    var m11 = M[q, q]

    # rot1 symmetrizes the 2x2 submatrix.
    var t = m00 + m11
    var d = m10 - m01

    var c1: RealScalar
    var s1: RealScalar
    if abs(d) < REAL_MIN:
        c1 = RealScalar(1)
        s1 = RealScalar(0)
    else:
        var u = t / d
        s1 = RealScalar(1) / sqrt(RealScalar(1) + u * u)
        c1 = u * s1

    # Apply rot1; result is symmetric, so only 3 distinct entries remain.
    var a00 = c1 * m00 + s1 * m10
    var a01 = c1 * m01 + s1 * m11
    var a11 = -s1 * m01 + c1 * m11

    var j_right = make_jacobi(a00, a01, a11)

    # j_left = rot1 * j_right^T.
    var jr_c = j_right.c
    var jr_s = j_right.s
    var j_left = JacobiRotation(c1 * jr_c + s1 * jr_s, s1 * jr_c - c1 * jr_s)

    return (j_left, j_right)
