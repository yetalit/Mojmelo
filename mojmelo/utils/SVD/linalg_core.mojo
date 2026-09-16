from std.memory import Layout, unsafe_memcpy, unsafe_memset_zero
from std.math import sqrt, ceil
from std.sys import CompilationTarget, simd_width_of
from std.algorithm import vectorize
from ..mojmelo_matmul import matmul as GEMM
from mojmelo.utils.algorithm import parallelize

comptime RealScalar = Float64

comptime REAL_EPSILON = 2.220446049250313e-16  # Float64 unit roundoff
comptime REAL_MIN = 2.2250738585072014e-308  # smallest positive normal Float64
comptime SQRT_REAL_MAX = sqrt(1.7976931348623157e308)

# ------------------------------------------------------------------------------
# ComputationInfo — mirrors Eigen::ComputationInfo.
# ------------------------------------------------------------------------------
@fieldwise_init
struct ComputationInfo(TrivialRegisterPassable):
    var value: Int

    def __eq__(self, other: ComputationInfo) -> Bool:
        return self.value == other.value

    def __ne__(self, other: ComputationInfo) -> Bool:
        return self.value != other.value

comptime INFO_SUCCESS = ComputationInfo(0)
comptime INFO_NUMERICAL_ISSUE = ComputationInfo(1)
comptime INFO_NO_CONVERGENCE = ComputationInfo(2)
comptime INFO_INVALID_INPUT = ComputationInfo(3)

comptime SIMD_WIDTH: Int = 4 * simd_width_of[RealScalar.DTYPE]() if CompilationTarget.is_apple_silicon() else 2 * simd_width_of[RealScalar.DTYPE]()
# ------------------------------------------------------------------------------
# Vec — a strided view/owner over a contiguous RealScalar buffer.
#
# Stands in for Eigen's ArrayXr / VectorXr / Block / VectorBlock / Map: a Vec
# produced by `.segment()`, `.head()`, `.tail()`, or a Mat's `.col()` /
# `.row()` / `.diagonal()` shares storage with its parent (via pointer +
# stride), so writes through the view land in the original buffer.
# ------------------------------------------------------------------------------
struct Vec(Sized):
    var data: Pointer[RealScalar, MutUntrackedOrigin]
    var n: Int
    var stride: Int
    var owns: Bool

    @always_inline
    def __init__(out self, n: Int):
        self.data = alloc(Layout[RealScalar](count=max(n, 1))).unsafe_leak()
        self.n = n
        self.stride = 1
        self.owns = True
        self.setZero()

    @always_inline
    def __init__(out self, data: Pointer[RealScalar, MutUntrackedOrigin], n: Int, stride: Int):
        self.data = data
        self.n = n
        self.stride = stride
        self.owns = False

    @always_inline
    def __deinit__(deinit self):
        if self.owns:
            self.data.unsafe_free()

    @always_inline
    def __len__(self) -> Int:
        return self.n

    @always_inline
    def __getitem__(self, i: Int) -> RealScalar:
        return self.data[unsafe_offset=i * self.stride]

    @always_inline
    def __setitem__(mut self, i: Int, v: RealScalar):
        self.data[unsafe_offset=i * self.stride] = v

    @always_inline
    def segment(self, start: Int, length: Int) -> Vec:
        return Vec(self.data.unsafe_offset(start * self.stride), length, self.stride)

    @always_inline
    def head(self, length: Int) -> Vec:
        return self.segment(0, length)

    @always_inline
    def tail(self, length: Int) -> Vec:
        return self.segment(self.n - length, length)

    @always_inline
    def copyFrom(self, other: Vec):
        if self.stride == 1 and other.stride == 1:
            unsafe_memcpy(dest=self.data, src=other.data, count=self.n)
        else:
            for i in range(self.n):
                self.data[unsafe_offset=i * self.stride] = other.data[unsafe_offset=i * other.stride]

    def setZero(self):
        if self.stride == 1:
            unsafe_memset_zero(self.data, self.n)
        else:
            for i in range(self.n):
                self.data[unsafe_offset=i * self.stride] = RealScalar(0)

    @always_inline
    def cwiseAbsMax(self) -> RealScalar:
        var m = 0.0
        if self.stride == 1:
            var data = self.data
            def findMax[simd_width: Int](idx: Int) {mut}:
                var max_in_vec = abs(data.unsafe_load[simd_width](idx)).reduce_max()
                if max_in_vec > m:
                    m = max_in_vec
            vectorize[SIMD_WIDTH](self.n, findMax)
        else:
            for i in range(self.n):
                var a = abs(self[i])
                if a > m:
                    m = a
        return m

    @always_inline
    def norm(self) -> RealScalar:
        var scale = self.cwiseAbsMax()

        if scale == 0:
            return 0

        var sum = 0.0
        if self.stride == 1:
            var data = self.data
            def sumSquares[simd_width: Int](idx: Int) {mut}:
                var x = data.unsafe_load[simd_width](idx) / scale
                sum += (x * x).reduce_add()
            vectorize[SIMD_WIDTH](self.n, sumSquares)
        else:
            for i in range(self.n):
                var x = self[i] / scale
                sum += x * x

        return scale * sqrt(sum)

    @always_inline
    def reverseInPlace(mut self):
        var i = 0
        var j = self.n - 1
        while i < j:
            swap(self[i], self[j])
            i += 1
            j -= 1

    @always_inline
    def stableNormalize(mut self):
        var nrm = self.norm()
        if nrm > 0:
            if self.stride == 1:
                def divide[simd_width: Int](idx: Int) {mut}:
                    self.data.unsafe_store[simd_width](idx, self.data.unsafe_load[simd_width](idx) / nrm)
                vectorize[SIMD_WIDTH](self.n, divide)
            else:
                for i in range(self.n):
                    self[i] = self[i] / nrm

    @staticmethod
    @always_inline
    def Unit(n: Int, k: Int) -> Vec:
        var v = Vec(n)
        v[k] = RealScalar(1)
        return v^

# ------------------------------------------------------------------------------
# IVec — the integer counterpart of Vec, standing in for Eigen's ArrayXi.
# ------------------------------------------------------------------------------
struct IVec(Sized):
    var data: Pointer[Int, MutUntrackedOrigin]
    var n: Int
    var owns: Bool

    @always_inline
    def __init__(out self, n: Int):
        self.data = alloc(Layout[Int](count=max(n, 1))).unsafe_leak()
        self.n = n
        self.owns = True
        unsafe_memset_zero(self.data, self.n)

    @always_inline
    def __init__(out self, data: Pointer[Int, MutUntrackedOrigin], n: Int):
        self.data = data
        self.n = n
        self.owns = False

    @always_inline
    def __deinit__(deinit self):
        if self.owns:
            self.data.unsafe_free()

    @always_inline
    def __len__(self) -> Int:
        return self.n

    @always_inline
    def __getitem__(self, i: Int) -> Int:
        return self.data[unsafe_offset=i]

    @always_inline
    def __setitem__(mut self, i: Int, v: Int):
        self.data[unsafe_offset=i] = v

    @always_inline
    def segment(self, start: Int, length: Int) -> IVec:
        return IVec(self.data.unsafe_offset(start), length)

# ------------------------------------------------------------------------------
# Mat — a column-major, strided view/owner over a RealScalar buffer.
# Stands in for Eigen's MatrixXr / Block. `.col()`, `.row()`, `.block()`, and
# `.diagonal()` all return comptimeing views, matching Eigen's Block semantics.
# ------------------------------------------------------------------------------
struct Mat(Copyable):
    var data: Pointer[RealScalar, MutUntrackedOrigin]
    var nrows: Int
    var ncols: Int
    var row_stride: Int
    var col_stride: Int
    var size: Int
    var owns: Bool

    @always_inline
    def __init__(out self, rows: Int, cols: Int):
        self.size = max(rows * cols, 1)
        self.data = alloc(Layout[RealScalar](count=self.size)).unsafe_leak()
        self.nrows = rows
        self.ncols = cols
        self.row_stride = 1
        self.col_stride = rows
        self.owns = True
        self.setZero()

    @always_inline
    def __init__(
        out self,
        data: Pointer[RealScalar, MutUntrackedOrigin],
        rows: Int,
        cols: Int,
        row_stride: Int,
        col_stride: Int,
    ):
        self.data = data
        self.nrows = rows
        self.ncols = cols
        self.row_stride = row_stride
        self.col_stride = col_stride
        self.size = self.nrows * self.ncols
        self.owns = False

    @always_inline
    def __init__(out self, *, copy: Self):
        self.nrows = copy.nrows
        self.ncols = copy.ncols
        self.size = max(copy.nrows * copy.ncols, 1)
        self.row_stride = 1
        self.col_stride = copy.nrows
        self.owns = True
        self.data = alloc(Layout[RealScalar](count=self.size)).unsafe_leak()
        for j in range(self.ncols):
            unsafe_memcpy(
                dest=self.data.unsafe_offset(j * self.col_stride),
                src=copy.data.unsafe_offset(j * copy.col_stride),
                count=self.nrows,
            )

    @always_inline
    def __deinit__(deinit self):
        if self.owns:
            self.data.unsafe_free()

    @always_inline
    def rows(self) -> Int:
        return self.nrows

    @always_inline
    def cols(self) -> Int:
        return self.ncols

    @always_inline
    def __getitem__(self, i: Int, j: Int) -> RealScalar:
        return self.data[unsafe_offset=i * self.row_stride + j * self.col_stride]

    @always_inline
    def __setitem__(mut self, i: Int, j: Int, v: RealScalar):
        self.data[unsafe_offset=i * self.row_stride + j * self.col_stride] = v

    @always_inline
    def block(self, i: Int, j: Int, rows: Int, cols: Int) -> Mat:
        var offset = i * self.row_stride + j * self.col_stride
        return Mat(self.data.unsafe_offset(offset), rows, cols, self.row_stride, self.col_stride)

    @always_inline
    def col(self, j: Int) -> Vec:
        var offset = j * self.col_stride
        return Vec(self.data.unsafe_offset(offset), self.nrows, self.row_stride)

    @always_inline
    def row(self, i: Int) -> Vec:
        var offset = i * self.row_stride
        return Vec(self.data.unsafe_offset(offset), self.ncols, self.col_stride)

    @always_inline
    def diagonal(self, k: Int = 0) -> Vec:
        # k == 0: main diagonal. k == -1: the first sub-diagonal.
        var offset: Int
        var length: Int
        if k >= 0:
            length = min(self.nrows, self.ncols - k)
            offset = k * self.col_stride
        else:
            length = min(self.nrows + k, self.ncols)
            offset = (-k) * self.row_stride
        var step = self.row_stride + self.col_stride
        return Vec(self.data.unsafe_offset(offset), length, step)

    @always_inline
    def copyFrom(self, other: Mat):
        for j in range(self.ncols):
            unsafe_memcpy(
                dest=self.data.unsafe_offset(j * self.col_stride),
                src=other.data.unsafe_offset(j * other.col_stride),
                count=self.nrows,
            )

    @always_inline
    def setZero(self):
        for j in range(self.ncols):
            unsafe_memset_zero(self.data.unsafe_offset(j * self.col_stride), self.nrows)

    @always_inline
    def cwiseAbsMax(self) -> RealScalar:
        var m = 0.0
        for j in range(self.ncols):
            var col_data = self.data.unsafe_offset(j * self.col_stride)
            def findMax[simd_width: Int](idx: Int) {mut}:
                var max_in_vec = abs(col_data.unsafe_load[simd_width](idx)).reduce_max()
                if max_in_vec > m:
                    m = max_in_vec
            vectorize[SIMD_WIDTH](self.nrows, findMax)
        return m

    @always_inline
    def swap_cols(mut self, a: Int, b: Int):
        var a_col = Vec(self.nrows)
        a_col.copyFrom(self.col(a))
        self.col(a).copyFrom(self.col(b))
        self.col(b).copyFrom(a_col)

    @staticmethod
    def zeros(rows: Int, cols: Int) -> Mat:
        return Mat(rows, cols)

# ------------------------------------------------------------------------------
# A handful of small free-function Vec/Mat helpers shared across files.
# ------------------------------------------------------------------------------
@always_inline
def swap_vecs(mut a: Vec, mut b: Vec):
    var tmp = Vec(len(a))
    tmp.copyFrom(a)
    a.copyFrom(b)
    b.copyFrom(tmp)

@always_inline
def vec_dot(a: Vec, b: Vec) -> RealScalar:
    var n = len(a)
    var sum = 0.0
    if a.stride == 1 and b.stride == 1:
        var ad = a.data
        var bd = b.data
        def dotChunk[simd_width: Int](idx: Int) {mut}:
            sum += (ad.unsafe_load[simd_width](idx) * bd.unsafe_load[simd_width](idx)).reduce_add()
        vectorize[SIMD_WIDTH](n, dotChunk)
    else:
        for i in range(n):
            sum += a[i] * b[i]
    return sum

@always_inline
def reverse_cols(mut m: Mat, count: Int):
    var i = 0
    var j = count - 1
    while i < j:
        m.swap_cols(i, j)
        i += 1
        j -= 1

@always_inline
def mat_transpose(a: Mat) -> Mat:
    var mat = Mat(a.cols(), a.rows())
    var col_stride = a.col_stride
    if mat.size < 98304:
        for i in range(a.rows()):
            var idx_row = i
            var tmpPtr = a.data.unsafe_offset(idx_row * a.row_stride)
    
            def convert[simd_width: Int](idx: Int) {mut}:
                mat.data.unsafe_store[simd_width](idx + idx_row * mat.rows(), tmpPtr.unsafe_strided_load[width=simd_width](col_stride))
                tmpPtr = tmpPtr.unsafe_offset(simd_width * col_stride)
            vectorize[SIMD_WIDTH](a.cols(), convert)
    else:
        @__parameter
        def p(i: Int):
            var idx_row = i
            var tmpPtr = a.data.unsafe_offset(idx_row * a.row_stride)
    
            def pconvert[simd_width: Int](idx: Int) {mut}:
                mat.data.unsafe_store[simd_width](idx + idx_row * mat.rows(), tmpPtr.unsafe_strided_load[width=simd_width](col_stride))
                tmpPtr = tmpPtr.unsafe_offset(simd_width * col_stride)
            vectorize[SIMD_WIDTH](a.cols(), pconvert)
        parallelize[p](a.rows())
    return mat^

@always_inline
def mat_scale(mut a: Mat, s: RealScalar):
    if a.size < 262144:
        def scalar_vectorize[simd_width: Int](idx: Int) {imm}:
            a.data.unsafe_store[simd_width](idx, a.data.unsafe_load[width=simd_width](idx) / s)
        vectorize[SIMD_WIDTH](a.size, scalar_vectorize)
    else:
        var n_vects = Int(ceil(a.size / SIMD_WIDTH))
        @__parameter
        def scalar_vectorize_parallelize(i: Int):
            var idx = i * SIMD_WIDTH
            a.data.unsafe_store[SIMD_WIDTH](idx, a.data.unsafe_load[width=SIMD_WIDTH](idx) / s)
        parallelize[scalar_vectorize_parallelize](n_vects)

@always_inline
def mat_identity(rows: Int, cols: Int) -> Mat:
    var m = Mat(rows, cols)
    var n = rows if rows < cols else cols

    var tmpPtr = m.data
    def convert[simd_width: Int](idx: Int) {mut}:
        tmpPtr.unsafe_strided_store[width=simd_width](1.0, (n + 1))
        tmpPtr = tmpPtr.unsafe_offset(simd_width * (n + 1))
    vectorize[SIMD_WIDTH](n, convert)

    return m^

@always_inline
def embed_topleft(dst: Mat, src: Mat, n: Int):
    for i in range(n):
        var col_dst = dst.col(i)
        var col_src = src.col(i)
        unsafe_memcpy(dest=col_dst.data, src=col_src.data, count=n)

@always_inline
def matmul(a: Mat, b: Mat) -> Mat:
    var result = Mat(a.rows(), b.cols())

    var A_T_rm = GEMM.Matrix[RealScalar.DTYPE](a.data, GEMM.MatLayout((a.cols(), a.rows()), (a.col_stride, a.row_stride)))
    var B_T_rm = GEMM.Matrix[RealScalar.DTYPE](b.data, GEMM.MatLayout((b.cols(), b.rows()), (b.col_stride, b.row_stride)))
    var C_T_rm = GEMM.Matrix[RealScalar.DTYPE](result.data, (b.cols(), a.rows()))

    GEMM.matmul(b.cols(), a.rows(), a.cols(), C_T_rm, B_T_rm, A_T_rm)

    return result^
