from std.algorithm import vectorize
from std.sys import simd_width_of
from mojmelo.utils.utils import _max_abs, _axpy, vec_scale, dot_config
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _eps[dtype: DType]() -> Float64:
    comptime if dtype == DType.float64:
        return 2.220446049250313e-16
    return 1.1920929e-07  # float32 / fallback

@always_inline
def _pivot_row[dtype: DType](a: Pointer[Scalar[dtype], MutUntrackedOrigin], n: Int, k: Int) -> Int:
    """Row index in k..n-1 with the largest |A[i, k]| (strided, scalar)."""
    var best_row = k
    var best = abs(a[unsafe_offset=k * n + k])
    for i in range(k + 1, n):
        var v = abs(a[unsafe_offset=i * n + k])
        if v > best:
            best = v
            best_row = i
    return best_row

@always_inline
def _swap[dtype: DType](var r1: Pointer[Scalar[dtype], MutUntrackedOrigin], var r2: Pointer[Scalar[dtype], MutUntrackedOrigin], count: Int):
    """Swap r1[0:count] with r2[0:count] (must not overlap)."""
    comptime W = simd_width_of[dtype]()

    def body[w: Int](idx: Int) {mut}:
        var u = r1.unsafe_load[w](idx)
        var v = r2.unsafe_load[w](idx)
        r1.unsafe_offset(idx).unsafe_store(v)
        r2.unsafe_offset(idx).unsafe_store(u)

    vectorize[W](count, body)


def _eliminate[
    dtype: DType
](a: Pointer[Scalar[dtype], MutUntrackedOrigin], x: Pointer[Scalar[dtype], MutUntrackedOrigin], n: Int, nrhs: Int) raises:
    """Reduce A to upper-triangular form with partial pivoting, applying the
    same row swaps and multipliers to X (the right-hand sides)."""
    var tol = (Float64(n) * _eps[dtype]()).cast[dtype]() * _max_abs[dtype](
        a, n * n
    )

    for k in range(n):
        var pr = _pivot_row[dtype](a, n, k)
        var pv = abs(a[unsafe_offset=pr * n + k])
        # `not (pv > tol)` also catches NaN.
        if not (pv > tol):
            raise Error("lu_solve: matrix is singular to working precision")

        if pr != k:
            # Columns < k of A are never read again, so swap from column k.
            _swap[dtype](
                a.unsafe_offset(k * n + k), a.unsafe_offset(pr * n + k), n - k
            )
            _swap[dtype](
                x.unsafe_offset(k * nrhs), x.unsafe_offset(pr * nrhs), nrhs
            )

        var inv = Scalar[dtype](1) / a[unsafe_offset=k * n + k]
        var width = n - k - 1
        var ak = a.unsafe_offset(k * n + k + 1)
        var xk = x.unsafe_offset(k * nrhs)
        for i in range(k + 1, n):
            var l = a[unsafe_offset=i * n + k] * inv
            _axpy[dtype](a.unsafe_offset(i * n + k + 1), ak, width, l)
            _axpy[dtype](x.unsafe_offset(i * nrhs), xk, nrhs, l)


def _back_substitute[
    dtype: DType
](a: Pointer[Scalar[dtype], MutUntrackedOrigin], x: Pointer[Scalar[dtype], MutUntrackedOrigin], n: Int, nrhs: Int):
    """Solve U X = Y in place, where U is the upper triangle of `a`."""
    if nrhs == 1:
        for ii in range(n):
            var i = n - 1 - ii
            var s = dot_config[dtype](
                a.unsafe_offset(i * n + i + 1), x.unsafe_offset(i + 1), n - i - 1
            )
            x[unsafe_offset=i] = (x[unsafe_offset=i] - s) / a[
                unsafe_offset=i * n + i
            ]
    else:
        for ii in range(n):
            var i = n - 1 - ii
            var xi = x.unsafe_offset(i * nrhs)
            for j in range(i + 1, n):
                _axpy[dtype](
                    xi, x.unsafe_offset(j * nrhs), nrhs, a[unsafe_offset=i * n + j]
                )
            vec_scale[dtype](
                xi, nrhs, Scalar[dtype](1) / a[unsafe_offset=i * n + i]
            )

# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def lu_solve[
    dtype: DType = DType.float64
](a: Pointer[Scalar[dtype], MutUntrackedOrigin], x: Pointer[Scalar[dtype], MutUntrackedOrigin], n: Int, nrhs: Int = 1) raises:
    """Solve A X = B in place.

    Args:
        a: (n x n) matrix, row-major. Overwritten by the solve.
        x: (n x nrhs) right-hand sides, row-major (a plain length-n vector when
           nrhs == 1). Holds B on entry and the solution X on exit.
        n: Matrix dimension.
        nrhs: Number of right-hand-side columns.

    Raises:
        If n or nrhs is not positive, or A is singular to working precision.
    """
    if n <= 0 or nrhs <= 0:
        raise Error("lu_solve: n and nrhs must be positive")

    _eliminate[dtype](a, x, n, nrhs)
    _back_substitute[dtype](a, x, n, nrhs)
