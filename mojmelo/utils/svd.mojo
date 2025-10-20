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
                for r in range(n):
                    if r == p or r == q:
                        continue
                    var Arp = A[r * n + p]
                    var Arq = A[r * n + q]
                    A[r * n + p] = c * Arp - s * Arq
                    A[p * n + r] = A[r * n + p]
                    A[r * n + q] = s * Arp + c * Arq
                    A[q * n + r] = A[r * n + q]

                # update eigenvector matrix V (columns p and q)
                var bp = UnsafePointer[Float64].alloc(n)
                var bq = UnsafePointer[Float64].alloc(n)
                var tmpPtr1 = V + p
                var tmpPtr2 = V + q
                @parameter
                fn read[simd_width: Int](idx: Int):
                    bp.store(idx, c * tmpPtr1.strided_load[width=simd_width](n) - s * tmpPtr2.strided_load[width=simd_width](n))
                    bq.store(idx, s * tmpPtr1.strided_load[width=simd_width](n) + c * tmpPtr2.strided_load[width=simd_width](n))
                    tmpPtr1 += simd_width * n
                    tmpPtr2 += simd_width * n
                vectorize[read, simd_width](n)
                tmpPtr1 = V + p
                tmpPtr2 = V + q
                @parameter
                fn write[simd_width: Int](idx: Int):
                    tmpPtr1.strided_store[width=simd_width](bp.load[width=simd_width](idx), n)
                    tmpPtr1 += simd_width * n
                    tmpPtr2.strided_store[width=simd_width](bq.load[width=simd_width](idx), n)
                    tmpPtr2 += simd_width * n
                vectorize[write, simd_width](n)
                bp.free()
                bq.free()
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

    var V_f = Matrix(V_full, n, n)['', sorted_indices]

    # V_full columns are eigenvectors (n x n), copy into Vout row r as transpose
    Vout = V_f.load_columns(k).T()
    # Build singular values S (k) and Vout (k x n) as the top-k eigenvectors
    for r in range(k):
        var lambda_ = eig[r]
        if lambda_ < 0 and abs(lambda_) < 1e-14:
            lambda_ = 0.0 # clamp tiny negative
        S[r] = math.sqrt(lambda_) if lambda_ > 0.0 else 0.0

    eig.free()
    ATA.free()

fn svd(A: Matrix, k: Int) raises -> Tuple[Matrix, Matrix]:
    var S = UnsafePointer[Float64].alloc(k)
    var V = Matrix(0, 0)
    svd_thin(A.height, A.width, k, S, V, (A.T() * A).cast_ptr[DType.float64]())
    return Matrix(S, 1, k), V^
