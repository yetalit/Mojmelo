from .mojmelo_matmul import matmul
from memory import memcpy, memset_zero
from algorithm import vectorize, parallelize
from sys import simd_width_of, CompilationTarget
import math
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import fill_indices_list

alias EPS = 1e-10
alias MAX_JACOBI_SWEEPS = 200
alias simd_width = 4 * simd_width_of[DType.float64]() if CompilationTarget.is_apple_silicon() else 2 * simd_width_of[DType.float64]()

fn jacobi_eigensystem(A_in: UnsafePointer[Float64], eig: UnsafePointer[Float64], V: UnsafePointer[Float64], n: Int):
    var A = UnsafePointer[Float64].alloc(n * n)
    memcpy(dest=A, src=A_in, count=n*n)

    # initialize V = I
    memset_zero(V, n*n)
    var tmpPtr = V
    @parameter
    fn eye[simd_width: Int](idx: Int):
        tmpPtr.strided_store[width=simd_width](1.0, (n + 1))
        tmpPtr += simd_width * (n + 1)
    vectorize[eye, simd_width](n)

    for sweep in range(MAX_JACOBI_SWEEPS):
        var max_off = 0.0
        for p in range(n-1):
            for q in range(p+1, n):
                var Apq = A[p * n + q]
                var a = abs(Apq)
                if a > max_off:
                    max_off = a
                if a <= EPS:
                    continue

                var App = A[p * n + p]
                var Aqq = A[q * n + q]
                var tau = (Aqq - App) / (2.0 * Apq)
                var t: Float64
                if (tau >= 0):
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                var c = 1.0 / math.sqrt(1.0 + t * t)
                var s = t * c

                # Apply rotation to A: only rows/cols p and q change
                A[p * n + p] = c*c*App - 2.0*c*s*Apq + s*s*Aqq
                A[q * n + q] = s*s*App + 2.0*c*s*Apq + c*c*Aqq
                A[p * n + q] = 0.0
                A[q * n + p] = 0.0

                # update other entries
                for r in range(p):
                    var Arp = A[r * n + p]
                    var Arq = A[r * n + q]
                    var Ap = c * Arp - s * Arq
                    var Aq = s * Arp + c * Arq
                    A[r * n + p] = Ap
                    A[p * n + r] = Ap
                    A[r * n + q] = Aq
                    A[q * n + r] = Aq
                
                for r in range(p+1, q):
                    var Arp = A[r * n + p]
                    var Arq = A[r * n + q]
                    var Ap = c * Arp - s * Arq
                    var Aq = s * Arp + c * Arq
                    A[r * n + p] = Ap
                    A[p * n + r] = Ap
                    A[r * n + q] = Aq
                    A[q * n + r] = Aq

                for r in range(q+1, n):
                    var Arp = A[r * n + p]
                    var Arq = A[r * n + q]
                    var Ap = c * Arp - s * Arq
                    var Aq = s * Arp + c * Arq
                    A[r * n + p] = Ap
                    A[p * n + r] = Ap
                    A[r * n + q] = Aq
                    A[q * n + r] = Aq

                # update eigenvector matrix V (columns p and q)
                @parameter
                fn column[simd_width: Int](idx: Int):
                    var Vp = (V+p*n).load[width=simd_width](idx)
                    var Vq = (V+q*n).load[width=simd_width](idx)
                    (V+p*n).store(idx, c * Vp - s * Vq)
                    (V+q*n).store(idx, s * Vp + c * Vq)
                vectorize[column, simd_width](n)
        if max_off <= EPS:
            break
        elif sweep == MAX_JACOBI_SWEEPS-1:
            print("WARN: Jacobi iterations no converge!")

    # extract eigenvalues (diagonal of A)
    tmpPtr = A
    @parameter
    fn diagonal[simd_width: Int](idx: Int):
        eig.store(idx, tmpPtr.strided_load[width=simd_width](n + 1))
        tmpPtr += simd_width * (n + 1)
    vectorize[diagonal, simd_width](n)

    A.free()

fn svd_thin(m: Int, n: Int, k: Int, S: UnsafePointer[Float64], mut Vout: Matrix, ATA: UnsafePointer[Float64]) raises:
    # Jacobi eigensolver on ATA to get eigenvalues (lambda) and eigenvectors (V_full)
    var eig = UnsafePointer[Float64].alloc(n)
    memset_zero(eig, n)
    var V_full = UnsafePointer[Float64].alloc(n*n)
    memset_zero(V_full, n*n)

    jacobi_eigensystem(ATA, eig, V_full, n)

    # Sort eigenpairs descending by eigenvalue
    var sorted_indices = fill_indices_list(n)
    @parameter
    fn cmp_fn(a: Float64, b: Float64) -> Bool:
        return a > b

    mojmelo.utils.sort.sort[cmp_fn](
        Span[
            Float64,
            origin_of(eig),
        ](ptr=eig, length=UInt(n)), sorted_indices.unsafe_ptr()
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

    eig.free()
    ATA.free()

fn svd(A: Matrix, k: Int) raises -> Tuple[Matrix, Matrix]:
    var A64 = A.cast_ptr[DType.float64]()
    var A64T = C_transpose(A, A64)

    var S = UnsafePointer[Float64].alloc(k)
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
fn C_transpose(A: Matrix, A64: UnsafePointer[Float64]) -> UnsafePointer[Float64]:
    var AT = UnsafePointer[Float64].alloc(A.size)
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
