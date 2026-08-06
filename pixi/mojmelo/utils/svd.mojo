import mojmelo.utils.sort as msort
from mojmelo_matmul import matmul
from std.memory import unsafe_memcpy, unsafe_memset_zero, Layout
from std.algorithm import vectorize
from mojmelo.utils.algorithm import parallelize
from std.sys import simd_width_of, CompilationTarget
import std.math as math
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import fill_indices_list

comptime EPS = 1e-13
comptime simd_width = 4 * simd_width_of[DType.float64]() if CompilationTarget.is_apple_silicon() else 2 * simd_width_of[DType.float64]()

def eigensystem(A: Pointer[Float64, MutUntrackedOrigin], eig: Pointer[Float64, MutUntrackedOrigin], V: Pointer[Float64, MutUntrackedOrigin], n: Int):
    unsafe_memcpy(dest=V, src=A, count=n*n)

    var e = alloc(Layout[Float64](count=n)).unsafe_leak()
    unsafe_memset_zero(e, n)

    # --- Householder reduction to tridiagonal ---
    for i in range(n - 1, 0, -1):
        var l = i - 1
        var scale = var h = 0.0
        if l > 0:
            for k in range(l+1):
                scale += abs(V[unsafe_offset=k * n + i])
            if scale == 0.0:
                e[unsafe_offset=i] = V[unsafe_offset=l * n + i]
            else:
                for k in range(l+1):
                    V[unsafe_offset=k * n + i] /= scale
                    h += V[unsafe_offset=k * n + i] * V[unsafe_offset=k * n + i]

                var f = V[unsafe_offset=l * n + i]
                var g = -math.sqrt(h) if f >= 0.0 else math.sqrt(h)
                e[unsafe_offset=i] = scale * g
                h -= f * g
                V[unsafe_offset=l * n + i] = f - g
                f = 0.0
                for j in range(l+1):
                    V[unsafe_offset=i * n + j] = V[unsafe_offset=j * n + i] / h
                    var s = 0.0
                    for k in range(j+1):
                        s += V[unsafe_offset=k * n + j] * V[unsafe_offset=k * n + i]
                    for k in range(j + 1, l+1):
                        s += V[unsafe_offset=j * n + k] * V[unsafe_offset=k * n + i]
                    e[unsafe_offset=j] = s / h
                    f += e[unsafe_offset=j] * V[unsafe_offset=j * n + i]

                var hh = f / (h + h)
                for j in range(l+1):
                    f = V[unsafe_offset=j * n + i]
                    e[unsafe_offset=j] -= hh * f
                    for k in range(j+1):
                        V[unsafe_offset=k * n + j] -= (f * e[unsafe_offset=k] + e[unsafe_offset=j] * V[unsafe_offset=k * n + i])

        else:
            e[unsafe_offset=i] = V[unsafe_offset=l * n + i]
        eig[unsafe_offset=i] = h

    eig[unsafe_offset=0] = 0.0
    e[unsafe_offset=0] = 0.0

    # --- Accumulate transformations ---
    for i in range(n):
        var l = i - 1
        if eig[unsafe_offset=i] != 0.0:
            for j in range(l+1):
                var s = 0.0
                for k in range(l+1):
                    s += V[unsafe_offset=k * n + i] * V[unsafe_offset=j * n + k]
                for k in range(l+1):
                    V[unsafe_offset=j * n + k] -= s * V[unsafe_offset=i * n + k]

        eig[unsafe_offset=i] = V[unsafe_offset=i * n + i]
        V[unsafe_offset=i * n + i] = 1.0
        for j in range(i):
            V[unsafe_offset=i * n + j] = V[unsafe_offset=j * n + i] = 0.0

    # --- Implicit QL algorithm ---
    for i in range(1, n):
        e[unsafe_offset=i - 1] = e[unsafe_offset=i]
    e[unsafe_offset=n - 1] = 0.0

    for l in range(n):
        var iter = 0
        while True:
            var m = l
            while m < n - 1:
                if abs(e[unsafe_offset=m]) <= EPS * (abs(eig[unsafe_offset=m]) + abs(eig[unsafe_offset=m + 1])):
                    break
                m += 1
            if m == l:
                break # converged
            if iter > 60:
                break # too many iterations, fallback
            iter += 1

            var g = (eig[unsafe_offset=l + 1] - eig[unsafe_offset=l]) / (2.0 * e[unsafe_offset=l])
            var r = math.hypot(g, 1.0)
            if g < 0:
                r = -r
            g = eig[unsafe_offset=m] - eig[unsafe_offset=l] + e[unsafe_offset=l] / (g + r)

            var s, c, p = 1.0, 1.0, 0.0
            for i in range(m - 1, l-1, -1):
                var f = s * e[unsafe_offset=i]
                var b = c * e[unsafe_offset=i]
                r = math.hypot(f, g)
                if r < 1e-300:
                    r = 1e-300
                e[unsafe_offset=i + 1] = r
                s = f / r
                c = g / r
                g = eig[unsafe_offset=i + 1] - p
                var t = (eig[unsafe_offset=i] - g) * s + 2.0 * c * b
                p = s * t
                eig[unsafe_offset=i + 1] = g + p
                g = c * t - b

                # update eigenvectors
                var n_full = n // simd_width
                var tail = n % simd_width
                for v_i in range(n_full):
                    var idx = v_i * simd_width

                    var tau = V.unsafe_offset((i + 1) * n).unsafe_load[width=simd_width](idx)
                    var Vki = V.unsafe_offset(i * n).unsafe_load[width=simd_width](idx)

                    V.unsafe_offset((i + 1) * n).unsafe_store(idx, s * Vki + c * tau)
                    V.unsafe_offset(i * n).unsafe_store(idx, c * Vki - s * tau)

                # Tail
                var tail_start = n_full * simd_width
                for t_i in range(tail):
                    var idx = tail_start + t_i

                    var tau = V.unsafe_offset((i + 1) * n)[unsafe_offset=idx]
                    var Vki = V.unsafe_offset(i * n)[unsafe_offset=idx]

                    V.unsafe_offset((i + 1) * n)[unsafe_offset=idx] = s * Vki + c * tau
                    V.unsafe_offset(i * n)[unsafe_offset=idx] = c * Vki - s * tau

            eig[unsafe_offset=l] -= p
            e[unsafe_offset=l] = g
            e[unsafe_offset=m] = 0.0

    e.unsafe_free()

def svd_thin(m: Int, n: Int, k: Int, S: Pointer[Float64, MutUntrackedOrigin], mut Vout: Matrix, ATA: Pointer[Float64, MutUntrackedOrigin]) raises:
    var eig = alloc(Layout[Float64](count=n)).unsafe_leak()
    unsafe_memset_zero(eig, n)
    var V_full = alloc(Layout[Float64](count=n*n)).unsafe_leak()

    eigensystem(ATA, eig, V_full, n)

    # Sort eigenpairs descending by eigenvalue
    var sorted_indices = fill_indices_list(n)
    @parameter
    def cmp_fn(a: Float64, b: Float64) -> Bool:
        return a > b

    msort.sort[cmp_fn](
        Span(unsafe_ptr=eig, length=n), Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(sorted_indices.unsafe_ptr()))
    )

    var V_f = Matrix(V_full, n, n, order='f')['', sorted_indices]

    # V_full columns are eigenvectors (n x n), copy into Vout row r as transpose
    Vout = V_f.load_columns(k)
    Vout.order = 'c'
    Vout = Vout.reshape(k, n)

    for r in range(n):
        var lambda_ = eig[unsafe_offset=r]
        if lambda_ < 0 and abs(lambda_) < 1e-14:
            lambda_ = 0.0 # clamp tiny negative
        S[unsafe_offset=r] = math.sqrt(lambda_) if lambda_ > 0.0 else 0.0
    
    ATA.unsafe_free()
    eig.unsafe_free()

def svd(A: Matrix, k: Int) raises -> Tuple[Matrix, Matrix]:
    var A64 = A.cast_ptr[DType.float64]()
    var A64T = C_transpose(A, A64)

    var S = alloc(Layout[Float64](count=A.width)).unsafe_leak()
    var V = Matrix(0, 0)

    var AT = matmul.Matrix[DType.float64](A64T, (A.width, A.height))
    var B = matmul.Matrix[DType.float64](A64, (A.height, A.width))
    var ATA = matmul.Matrix[DType.float64]((A.width, A.width))
    unsafe_memset_zero(ATA.data, A.width * A.width)
    matmul.matmul(A.width, A.height, A.width, ATA, AT, B)
    A64.unsafe_free()
    A64T.unsafe_free()

    svd_thin(A.height, A.width, k, S, V, ATA.data)
    return Matrix(S, 1, A.width), V^

@always_inline
def C_transpose(A: Matrix, A64: Pointer[Float64, MutUntrackedOrigin]) -> Pointer[Float64, MutUntrackedOrigin]:
    var AT = alloc(Layout[Float64](count=A.size)).unsafe_leak()
    var height = A.height
    var width = A.width
    if A.size < 98304:
        for i in range(A.width):
            var idx_col = i
            var tmpPtr = A64.unsafe_offset(idx_col)
            def convert[simd_width: Int](idx: Int) {mut}:
                AT.unsafe_store(idx + idx_col * height, tmpPtr.unsafe_strided_load[width=simd_width](width))
                tmpPtr = tmpPtr.unsafe_offset(simd_width * width)
            vectorize[simd_width](A.height, convert)
    else:
        @parameter
        def p(i: Int):
            var idx_col = i
            var tmpPtr = A64.unsafe_offset(idx_col)
            def pconvert[simd_width: Int](idx: Int) {mut}:
                AT.unsafe_store(idx + idx_col * height, tmpPtr.unsafe_strided_load[width=simd_width](width))
                tmpPtr = tmpPtr.unsafe_offset(simd_width * width)
            vectorize[simd_width](A.height, pconvert)
        parallelize[p](A.width)
    return AT
