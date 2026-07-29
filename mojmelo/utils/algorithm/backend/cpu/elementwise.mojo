# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""CPU implementations of elementwise functions."""

from std.math import ceildiv

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import IndexList

from std.gpu.host import DeviceContext
from std.algorithm.backend.vectorize import vectorize

from .parallelize import _get_num_workers, sync_parallelize


@always_inline
def _get_start_indices_of_nth_subvolume[
    rank: Int, //, subvolume_rank: Int = 1
](n: Int, shape: IndexList[rank, ...], out res: type_of(shape)):
    """Converts a flat index into the starting ND indices of the nth subvolume
    with rank `subvolume_rank`.

    For example:
        - `_get_start_indices_of_nth_subvolume[0](n, shape)` will return
        the starting indices of the nth element in shape.
        - `_get_start_indices_of_nth_subvolume[1](n, shape)` will return
        the starting indices of the nth row in shape.
        - `_get_start_indices_of_nth_subvolume[2](n, shape)` will return
        the starting indices of the nth horizontal slice in shape.

    The ND indices will iterate from right to left. I.E

        shape = (20, 5, 2, N)
        _get_start_indices_of_nth_subvolume[1](1, shape) = (0, 0, 1, 0)
        _get_start_indices_of_nth_subvolume[1](5, shape) = (0, 2, 1, 0)
        _get_start_indices_of_nth_subvolume[1](50, shape) = (5, 0, 0, 0)
        _get_start_indices_of_nth_subvolume[1](56, shape) = (5, 1, 1, 0)

    Parameters:
        rank: The rank of the ND index.
        subvolume_rank: The rank of the subvolume under consideration.

    Args:
        n: The flat index to convert (the nth subvolume to retrieve).
        shape: The shape of the ND space we are converting into.

    Returns:
        Constructed ND-index.
    """

    comptime assert (
        subvolume_rank <= rank
    ), "subvolume rank cannot be greater than indices rank"
    comptime assert subvolume_rank >= 0, "subvolume rank must be non-negative"

    comptime IntType = type_of(shape)._int_type

    # fast impls for common cases
    comptime if rank == 2 and subvolume_rank == 1:
        return {n, 0}

    comptime if rank - 1 == subvolume_rank:
        res = {0}
        res[0] = n
        return

    comptime if rank == subvolume_rank:
        return {0}

    res = {}
    # If the index type is unsigned, be sure to use unsigned div/mod operations.
    var curr_index = IntType(n)

    comptime for i in reversed(range(1, rank - subvolume_rank)):
        curr_index, res.data[i] = divmod(curr_index, IntType(shape.get[i]()))

    res.data[0] = curr_index


# ===-----------------------------------------------------------------------===#
# Elementwise CPU implementations
# ===-----------------------------------------------------------------------===#


@always_inline
def _elementwise_impl_cpu[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
    *,
    trace_description: StaticString,
](func: FuncType, *, shape: Coord, ctx: Optional[DeviceContext] = None,):
    """Dispatches elementwise execution on CPU to the 1D or ND implementation
    based on the rank of the input shape.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.
        trace_description: Description of the trace.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """

    comptime impl = _elementwise_impl_cpu_1d if shape.rank == 1 else _elementwise_impl_cpu_nd
    impl[simd_width](func, shape, ctx)


@always_inline
def _elementwise_impl_cpu_1d[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
](func: FuncType, shape: Coord, ctx: Optional[DeviceContext] = None,):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """
    comptime assert shape.rank == 1, "Specialization for 1D"

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    var problem_size = shape.product()

    var num_workers = _get_num_workers(problem_size, ctx=ctx)
    var chunk_size = ceildiv(problem_size, num_workers)

    @always_inline
    def task_func(i: Int) {imm}:
        var start_offset = i * chunk_size
        var end_offset = min((i + 1) * chunk_size, problem_size)
        var len = end_offset - start_offset

        @always_inline
        def func_wrapper[
            simd_width: Int
        ](idx: Int) {imm start_offset, imm func,}:
            var offset = start_offset + idx
            func[simd_width](Coord(offset))

        vectorize[simd_width, unroll_factor=unroll_factor](len, func_wrapper)

    sync_parallelize(task_func, num_workers, ctx)


@always_inline
def _elementwise_impl_cpu_nd[
    simd_width: Int,
    FuncType: def[width: Int, alignment: Int = 1](Coord) -> None,
](func: FuncType, shape: Coord, ctx: Optional[DeviceContext] = None,):
    """Executes `func[width, rank](indices)`, possibly using sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns
    when all sub-tasks have completed.

    Parameters:
        simd_width: The SIMD vector width to use.
        FuncType: The body function type.

    Args:
        func: The closure carrying the captured state of the body function.
        shape: The shape of the buffer.
        ctx: Optional CPU DeviceContext to execute the tasks on.
    """
    comptime assert shape.rank > 1, "Specialization for ND where N > 1"
    comptime rank = shape.rank

    # If we know we won't do any work, return early
    if shape[rank - 1].value() == 0:
        return

    comptime unroll_factor = 8  # TODO: Comeup with a cost heuristic.

    # Strategy: we parallelize over all dimensions except the innermost and
    # vectorize over the innermost dimension. We unroll the innermost dimension
    # by a factor of unroll_factor.

    # Compute the number of workers to allocate based on ALL work, not just
    # the dimensions we split across.
    var total_size = shape.product()

    var num_workers = _get_num_workers(total_size, ctx=ctx)
    var parallelism_size = total_size // SIMDLength(shape[rank - 1].value())
    var chunk_size = ceildiv(parallelism_size, num_workers)

    @always_inline
    def task_func(i: Int) {imm}:
        var start_parallel_offset = i * chunk_size
        var end_parallel_offset = min((i + 1) * chunk_size, parallelism_size)

        var len = end_parallel_offset - start_parallel_offset
        if len <= 0:
            return

        var indices = IndexList[rank]()

        @always_inline
        def func_wrapper_nd[
            simd_width: Int
        ](idx: Int) {mut indices, imm func, imm}:
            indices[rank - 1] = idx
            func[simd_width](Coord(indices.canonicalize()))

        for parallel_offset in range(
            start_parallel_offset, end_parallel_offset
        ):
            indices = _get_start_indices_of_nth_subvolume(
                parallel_offset, coord_to_index_list(shape)
            )

            # We vectorize over the innermost dimension.
            vectorize[simd_width, unroll_factor=unroll_factor](
                SIMDLength(shape[rank - 1].value()), func_wrapper_nd
            )

    sync_parallelize(task_func, num_workers, ctx)
