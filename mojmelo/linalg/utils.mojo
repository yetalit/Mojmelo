from mojmelo.linalg.Matrix import Matrix
from std.sys import simd_width_of
from std import math
from std.memory import Layout
from std.algorithm import vectorize
from mojmelo.utils.algorithm import parallelize

# ===-----------------------------------------------------------------------===#
# argn
# ===-----------------------------------------------------------------------===#

def argn[is_max: Bool](input: Matrix, output: Matrix):
    comptime simd_width = simd_width_of[DType.float32]()
    var axis_size = input.size
    var input_stride = input.size
    comptime output_stride = 1
    comptime chunk_size = 1
    comptime parallel_size = 1

    @__copy_capture(
        axis_size, chunk_size, output_stride, input_stride, parallel_size
    )

    @__parameter
    @always_inline
    def cmpeq[
        dtype: DType, simd_width: SIMDLength
    ](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
        DType.bool, simd_width
    ]:
        comptime if is_max:
            return a.le(b)
        else:
            return a.ge(b)

    @__parameter
    @always_inline
    def cmp[
        dtype: DType, simd_width: SIMDLength
    ](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
        DType.bool, simd_width
    ]:
        comptime if is_max:
            return a.lt(b)
        else:
            return a.gt(b)

    # iterate over flattened axes
    comptime start = 0
    comptime end = 1
    for i in range(start, end):
        var input_offset = i * input_stride
        var output_offset = i * output_stride
        var input_dim_ptr = input.data.unsafe_offset(input_offset)
        var output_dim_ptr = output.data.unsafe_offset(output_offset)
        var global_val: Float32

        # initialize limits
        comptime if is_max:
            global_val = Float32.MIN
        else:
            global_val = Float32.MAX

        # initialize vector of maximal/minimal values
        var global_values: SIMD[DType.float32, simd_width]
        if axis_size < simd_width:
            global_values = global_val
        else:
            global_values = input_dim_ptr.unsafe_load[width=simd_width]()

        # iterate over values evenly divisible by simd_width
        var indices = math.iota[DType.float32, simd_width]()
        var global_indices = indices
        var last_simd_index = math.align_down(axis_size, simd_width)
        for j in range(simd_width, last_simd_index, simd_width):
            var curr_values = input_dim_ptr.unsafe_load[width=simd_width](j)
            indices += Float32(simd_width)

            var mask = cmpeq(curr_values, global_values)
            global_indices = mask.select(global_indices, indices)
            global_values = mask.select(global_values, curr_values)

        comptime if is_max:
            global_val = global_values.reduce_max()
        else:
            global_val = global_values.reduce_min()

        # Check trailing indices.
        var idx = Float32(0)
        var found_min: Bool = False
        for j in range(last_simd_index, axis_size, 1):
            var elem = input_dim_ptr.unsafe_load(j)
            if cmp(global_val, elem):
                global_val = elem
                idx = Float32(j)
                found_min = True

        # handle the case where min wasn't in trailing values
        if not found_min:
            var matching = global_values.eq(global_val)
            var min_indices = matching.select(
                global_indices, Float32.MAX
            )
            idx = min_indices.reduce_min()
        output_dim_ptr[] = idx

# ===----------------------------------------------------------------------===#

@always_inline
def add[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a + b

@always_inline
def sub[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a - b

@always_inline
def mul[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a * b

@always_inline
def div[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return a / b

@always_inline
def eq[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.eq(b)

@always_inline
def ne[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.ne(b)

@always_inline
def gt[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.gt(b)

@always_inline
def ge[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.ge(b)

@always_inline
def lt[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.lt(b)

@always_inline
def le[dtype: DType, width: Int](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[DType.bool, width]:
    return a.le(b)

@always_inline
def cast[src: DType, des: DType, width: Int](data: Pointer[Scalar[src], MutUntrackedOrigin], size: Int) -> Pointer[Scalar[des], MutUntrackedOrigin]:
    var ptr = alloc(Layout[Scalar[des]](count=size)).unsafe_leak()
    if size < 262144:

        def matrix_vectorize[simd_width: Int](idx: Int) {imm}:
            ptr.unsafe_store(idx, data.unsafe_load[width=simd_width](idx).cast[des]())
        vectorize[width](size, matrix_vectorize)
    else:
        var n_vects = size // width
        @__parameter
        def matrix_vectorize_parallelize(i: Int):
            var idx = i * width
            ptr.unsafe_store(idx, data.unsafe_load[width=width](idx).cast[des]())
        parallelize[matrix_vectorize_parallelize](n_vects)
        var ptrTail = ptr.unsafe_offset(n_vects * width)
        var dataTail = data.unsafe_offset(n_vects * width)
        def tail[simd_width: Int](idx: Int) {imm}:
            ptrTail.unsafe_store(idx, dataTail.unsafe_load[width=simd_width](idx).cast[des]())
        vectorize[width](size % width, tail)
    return ptr

@always_inline
def elemwise_scalar[
    dtype: DType,
    width: Int,
    func: def[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width]) thin->SIMD[dtype, width]
](
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin],
    src: Pointer[Scalar[dtype], MutUntrackedOrigin],
    count: Int,
    s: Scalar[dtype],
):
    if count < 262144:
        def scalar_vectorize[simd_width: Int](idx: Int) {imm}:
            dst.unsafe_store(idx, func(src.unsafe_load[width=simd_width](idx), s))
        vectorize[width](count, scalar_vectorize)
    else:
        var n_vects = count // width
        @__parameter
        def scalar_vectorize_parallelize(i: Int):
            var idx = i * width
            dst.unsafe_store(idx, func(src.unsafe_load[width=width](idx), s))
        parallelize[scalar_vectorize_parallelize](n_vects)
        var dstTail = dst.unsafe_offset(n_vects * width)
        var srcTail = src.unsafe_offset(n_vects * width)
        def tail[simd_width: Int](idx: Int) {imm}:
            dstTail.unsafe_store(idx, func(srcTail.unsafe_load[width=simd_width](idx), s))
        vectorize[width](count % width, tail)

@always_inline
def elemwise_matrix[
    dtype: DType,
    width: Int,
    func: def[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width]) thin->SIMD[dtype, width]
](
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin],
    lhs: Pointer[Scalar[dtype], MutUntrackedOrigin],
    rhs: Pointer[Scalar[dtype], MutUntrackedOrigin],
    count: Int,
):
    if count < 262144:
        def matrix_vectorize[simd_width: Int](idx: Int) {imm}:
            dst.unsafe_store(idx, func(lhs.unsafe_load[width=simd_width](idx), rhs.unsafe_load[width=simd_width](idx)))
        vectorize[width](count, matrix_vectorize)
    else:
        var n_vects = count // width
        @__parameter
        def matrix_vectorize_parallelize(i: Int):
            var idx = i * width
            dst.unsafe_store(idx, func(lhs.unsafe_load[width=width](idx), rhs.unsafe_load[width=width](idx)))
        parallelize[matrix_vectorize_parallelize](n_vects)
        var dstTail = dst.unsafe_offset(n_vects * width)
        var lhsTail = lhs.unsafe_offset(n_vects * width)
        var rhsTail = rhs.unsafe_offset(n_vects * width)
        def tail[simd_width: Int](idx: Int) {imm}:
            dstTail.unsafe_store(idx, func(lhsTail.unsafe_load[width=simd_width](idx), rhsTail.unsafe_load[width=simd_width](idx)))
        vectorize[width](count % width, tail)

@always_inline
def _max_abs[dtype: DType, width: Int](var p: Pointer[Scalar[dtype], MutUntrackedOrigin], count: Int) -> Scalar[dtype]:
    var m = Scalar[dtype](0)

    def findMax[simd_width: Int](idx: Int) {mut}:
        var max_in_vec = abs(p.unsafe_load[simd_width](idx)).reduce_max()
        if max_in_vec > m:
            m = max_in_vec

    vectorize[width](count, findMax)
    return m

@always_inline
def _axpy[
    dtype: DType,
    width: Int
](
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin],
    src: Pointer[Scalar[dtype], MutUntrackedOrigin],
    count: Int,
    alpha: Scalar[dtype],
):
    """dst[0:count] -= alpha * src[0:count] (ranges must not overlap)."""
    def body[w: Int](idx: Int) {imm}:
        dst.unsafe_offset(idx).unsafe_store(
            dst.unsafe_load[w](idx) - SIMD[dtype, w](alpha) * src.unsafe_load[w](idx)
        )

    vectorize[width](count, body)

@always_inline
def dot_unrolled[
    dtype: DType,
    width: Int
](pa: Pointer[Scalar[dtype], MutUntrackedOrigin], pb: Pointer[Scalar[dtype], MutUntrackedOrigin], count: Int) -> Scalar[dtype]:
    """Dot product with 4 independent SIMD accumulators, so the FMA latency
    chain of a single accumulator doesn't bound throughput. `vectorize` handles
    whatever is left after the unrolled main loop."""
    var a0 = SIMD[dtype, width](0)
    var a1 = SIMD[dtype, width](0)
    var a2 = SIMD[dtype, width](0)
    var a3 = SIMD[dtype, width](0)
    var j = 0
    while j + 4 * width <= count:
        a0 += pa.unsafe_load[width](j) * pb.unsafe_load[width](j)
        a1 += pa.unsafe_load[width](j + width) * pb.unsafe_load[width](j + width)
        a2 += pa.unsafe_load[width](j + 2 * width) * pb.unsafe_load[width](j + 2 * width)
        a3 += pa.unsafe_load[width](j + 3 * width) * pb.unsafe_load[width](j + 3 * width)
        j += 4 * width

    var acc = (a0 + a1) + (a2 + a3)
    var tail = Scalar[dtype](0)
    var qa = pa.unsafe_offset(j)
    var qb = pb.unsafe_offset(j)

    def body[w: Int](idx: Int) {mut}:
        acc += qa.unsafe_load[width](idx) * qb.unsafe_load[width](idx)

    vectorize[width](count - j, body)
    return acc.reduce_add() + tail
