from .mojmelo_matmul import matmul
from memory import memcpy, memset_zero
from algorithm import vectorize, parallelize
from sys import simd_width_of, CompilationTarget
import math
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import fill_indices_list

comptime EPS = 1e-13
comptime simd_width = 4 * simd_width_of[DType.float64]() if CompilationTarget.is_apple_silicon() else 2 * simd_width_of[DType.float64]()

fn eigensystem(A: UnsafePointer[Float64, MutAnyOrigin], eig: UnsafePointer[Float64, MutAnyOrigin], V: UnsafePointer[Float64, MutAnyOrigin], n: Int):
    memcpy(dest=V, src=A, count=n*n)

    var e = alloc[Float64](n)
    memset_zero(e, n)

    # --- Householder reduction to tridiagonal ---
    for i in range(n - 1, 0, -1):
        var l = i - 1
        var scale = 0.0; h = 0.0
        if l > 0:
            for k in range(l+1):
                scale += abs(V[k * n + i])
            if scale == 0.0:
                e[i] = V[l * n + i]
            else:
                for k in range(l+1):
                    V[k * n + i] /= scale
                    h += V[k * n + i] * V[k * n + i]

                var f = V[l * n + i]
                var g = -math.sqrt(h) if f >= 0.0 else math.sqrt(h)
                e[i] = scale * g
                h -= f * g
                V[l * n + i] = f - g
                f = 0.0
                for j in range(l+1):
                    V[i * n + j] = V[j * n + i] / h
                    var s = 0.0
                    for k in range(j+1):
                        s += V[k * n + j] * V[k * n + i]
                    for k in range(j + 1, l+1):
                        s += V[j * n + k] * V[k * n + i]
                    e[j] = s / h
                    f += e[j] * V[j * n + i]

                var hh = f / (h + h)
                for j in range(l+1):
                    f = V[j * n + i]
                    e[j] -= hh * f
                    for k in range(j+1):
                        V[k * n + j] -= (f * e[k] + e[j] * V[k * n + i])

        else:
            e[i] = V[l * n + i]
        eig[i] = h

    eig[0] = 0.0
    e[0] = 0.0

    # --- Accumulate transformations ---
    for i in range(n):
        var l = i - 1
        if eig[i] != 0.0:
            for j in range(l+1):
                var s = 0.0
                for k in range(l+1):
                    s += V[k * n + i] * V[j * n + k]
                for k in range(l+1):
                    V[j * n + k] -= s * V[i * n + k]

        eig[i] = V[i * n + i]
        V[i * n + i] = 1.0
        for j in range(i):
            V[i * n + j] = V[j * n + i] = 0.0

    # --- Implicit QL algorithm ---
    for i in range(1, n):
        e[i - 1] = e[i]
    e[n - 1] = 0.0

    for l in range(n):
        var iter = 0
        while True:
            var m = l
            while m < n - 1:
                if abs(e[m]) <= EPS * (abs(eig[m]) + abs(eig[m + 1])):
                    break
                m += 1
            if m == l:
                break # converged
            if iter > 60:
                break # too many iterations, fallback
            iter += 1

            var g = (eig[l + 1] - eig[l]) / (2.0 * e[l])
            var r = math.hypot(g, 1.0)
            if g < 0:
                r = -r
            g = eig[m] - eig[l] + e[l] / (g + r)

            var s = 1.0; c = 1.0; p = 0.0
            for i in range(m - 1, l-1, -1):
                var f = s * e[i]
                var b = c * e[i]
                r = math.hypot(f, g)
                if r < 1e-300:
                    r = 1e-300
                e[i + 1] = r
                s = f / r
                c = g / r
                g = eig[i + 1] - p
                var t = (eig[i] - g) * s + 2.0 * c * b
                p = s * t
                eig[i + 1] = g + p
                g = c * t - b

                # update eigenvectors
                @parameter
                fn column[simd_width: Int](idx: Int):
                    var tau = (V+(i + 1)*n).load[width=simd_width](idx)
                    var Vki = (V+i*n).load[width=simd_width](idx)
                    (V+(i + 1)*n).store(idx, s * Vki + c * tau)
                    (V+i*n).store(idx, c * Vki - s * tau)
                vectorize[column, simd_width](n)

            eig[l] -= p
            e[l] = g
            e[m] = 0.0

    e.free()

fn svd_thin(m: Int, n: Int, k: Int, S: UnsafePointer[Float64, MutAnyOrigin], mut Vout: Matrix, ATA: UnsafePointer[Float64, MutAnyOrigin]) raises:
    # Jacobi eigensolver on ATA to get eigenvalues (lambda) and eigenvectors (V_full)
    var eig = alloc[Float64](n)
    memset_zero(eig, n)
    var V_full = alloc[Float64](n*n)

    eigensystem(ATA, eig, V_full, n)

    # Sort eigenpairs descending by eigenvalue
    var sorted_indices = fill_indices_list(n)
    @parameter
    fn cmp_fn(a: Float64, b: Float64) -> Bool:
        return a > b

    mojmelo.utils.sort.sort[cmp_fn](
        Span[
            Float64,
            MutAnyOrigin,
        ](ptr=eig, length=n), sorted_indices.unsafe_ptr()
    )

    var V_f = Matrix(V_full, n, n, order='f')['', sorted_indices]

    # V_full columns are eigenvectors (n x n), copy into Vout row r as transpose
    Vout = V_f.load_columns(k)
    Vout.order = 'c'
    Vout = Vout.reshape(k, n)
    # Build singular values S (k) and Vout (k x n) as the top-k eigenvectors
    for r in range(k):
        var lambda_ = eig[r]
        if lambda_ < 0 and abs(lambda_) < 1e-14:
            lambda_ = 0.0 # clamp tiny negative
        S[r] = math.sqrt(lambda_) if lambda_ > 0.0 else 0.0
    
    ATA.free()
    eig.free()

fn svd(A: Matrix, k: Int) raises -> Tuple[Matrix, Matrix]:
    var A64 = A.cast_ptr[DType.float64]()
    var A64T = C_transpose(A, A64)

    var S = alloc[Float64](k)
    var V = Matrix(0, 0)

    var AT = matmul.Matrix[DType.float64](A64T, (A.width, A.height))
    var B = matmul.Matrix[DType.float64](A64, (A.height, A.width))
    var ATA = matmul.Matrix[DType.float64]((A.width, A.width))
    memset_zero(ATA.data, A.width * A.width)
    matmul.matmul(A.width, A.height, A.width, ATA, AT, B)
    A64.free()
    A64T.free()
    
    svd_thin(A.height, A.width, k, S, V, ATA.data)
    return Matrix(S, 1, k), V^

@always_inline
fn C_transpose(A: Matrix, A64: UnsafePointer[Float64, MutAnyOrigin]) -> UnsafePointer[Float64, MutAnyOrigin]:
    var AT = alloc[Float64](A.size)
    if A.size < 98304:
        for idx_col in range(A.width):
            var tmpPtr = A64 + idx_col
            @parameter
            fn convert[simd_width: Int](idx: Int):
                AT.store(idx + idx_col * A.height, tmpPtr.strided_load[width=simd_width](A.width))
                tmpPtr += simd_width * A.width
            vectorize[convert, simd_width](A.height)
    else:
        @parameter
        fn p(idx_col: Int):
            var tmpPtr = A64 + idx_col
            @parameter
            fn pconvert[simd_width: Int](idx: Int):
                AT.store(idx + idx_col * A.height, tmpPtr.strided_load[width=simd_width](A.width))
                tmpPtr += simd_width * A.width
            vectorize[pconvert, simd_width](A.height)
        parallelize[p](A.width)
    return AT
