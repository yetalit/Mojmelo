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

from std._plugin import CurrentPlugin, PluginForTarget
from std.collections.string.string_slice import get_static_string
from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.gpu.host.info import is_cpu, is_gpu
from std.runtime.tracing import Trace, TraceLevel, get_safe_task_id, trace_arg
from std.sys.info import CompilationTarget, _accelerator_arch

from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import Index, IndexList

# Re-export CPU implementations.
from .backend.cpu import (
    _elementwise_impl_cpu,
    _get_num_workers,
    parallelize,
    parallelize_over_rows,
    sync_parallelize,
)

# ===-----------------------------------------------------------------------===#
# Elementwise
# ===-----------------------------------------------------------------------===#


@always_inline
def elementwise[
    func: def[width: Int, alignment: Int = 1](Coord) capturing[_] -> None,
    simd_width: Int,
    *,
    target: StaticString = "cpu",
    _trace_description: StaticString = "elementwise",
](shape: Int, context: DeviceContext) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        func: The body function.
        simd_width: The SIMD vector width to use.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.

    Raises:
        If the operation fails.
    """

    elementwise[
        func,
        simd_width=simd_width,
        target=target,
        _trace_description=_trace_description,
    ](Coord(shape), context)


@always_inline
def elementwise[
    func: def[width: Int, alignment: Int = 1](Coord) capturing[_] -> None,
    simd_width: Int,
    *,
    target: StaticString = "cpu",
    _trace_description: StaticString = "elementwise",
](shape: Coord, context: DeviceContext) raises:
    """Executes `func[width, rank](indices)`, possibly as sub-tasks, for a
    suitable combination of width and indices so as to cover shape. Returns when
    all sub-tasks have completed.

    Parameters:
        func: The body function.
        simd_width: The SIMD vector width to use.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        shape: The shape of the buffer.
        context: The device context to use.

    Raises:
        If the operation fails.
    """

    def func_unified[width: Int, alignment: Int = 1](indices: Coord) {}:
        func[width, alignment](indices)

    _elementwise_impl[
        simd_width,
        target=target,
        trace_description=_trace_description,
    ](func_unified, shape, context)


@always_inline
def elementwise[
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    simd_width: Int,
    *,
    target: StaticString = "cpu",
    _trace_description: StaticString = "elementwise",
](func: FuncType, shape: Coord, context: DeviceContext) raises:
    """Unified-closure entry point for `elementwise` (DeviceContext).

    Accepts a parametric body (already in
    unified-closure form, with explicit captures) and dispatches to
    `_elementwise_impl`. `rank` and `FuncType` are inferred from the
    runtime `shape` and `func` arguments, so `simd_width` is the only
    explicit positional parameter — callers can write
    `elementwise[N](func, shape, ctx)`.

    Parameters:
        FuncType: A parametric callable taking
            `IndexList[rank]` and template parameters `width`, `rank`,
            `alignment`.
        simd_width: The SIMD vector width to use.
        target: The target to run on.
        _trace_description: Description of the trace.

    Args:
        func: The body closure value.
        shape: The shape of the buffer.
        context: The device context to use.

    Raises:
        If the operation fails.
    """
    _elementwise_impl[
        simd_width,
        target=target,
        trace_description=_trace_description,
    ](func, shape, context)


@fieldwise_init
struct _IndexListToCoordAdapter[
    rank: Int,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, rank: Int, alignment: Int = 1](IndexList[rank]) -> None,
](
    ImplicitlyCopyable,
    RegisterPassable,
    def[width: Int, alignment: Int = 1](Coord) -> None,
):
    """Adapts an `IndexList`-taking function to a `Coord`-taking callable.

    Bridges the `IndexList`-based elementwise body convention to the
    `Coord`-based GPU kernel interface required by `_elementwise_impl_gpu`.

    TODO(MOCO-4071): Use a closure instead, using a struct avoids a generic
    `lit.closure.init` with parametric witnesses in the MOGG package that the
    package loader cannot resolve.

    Parameters:
        rank: The rank of the index space.
        FuncType: The wrapped function type.
    """

    var func: Self.FuncType

    @always_inline
    def __call__[width: Int, alignment: Int = 1](self, coords: Coord):
        self.func[width, Self.rank, alignment](
            rebind[IndexList[Self.rank]](coord_to_index_list(coords))
        )


@fieldwise_init
struct _CoordToIndexListAdapter[
    rank: Int,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
](
    ImplicitlyCopyable,
    RegisterPassable,
    def[width: Int, rank: Int, alignment: Int = 1](IndexList[rank]) -> None,
):
    """Adapts a `Coord`-taking function to an `IndexList`-taking callable.

    Bridges the `Coord`-based elementwise body convention to the
    `IndexList`-based plugin entry points required by
    `CurrentPlugin.elementwise_fn`.

    TODO(MOCO-4071): Use a closure instead, using a struct avoids a generic
    `lit.closure.init` with parametric witnesses in the MOGG package that the
    package loader cannot resolve.

    Parameters:
        rank: The rank of the index space.
        FuncType: The wrapped function type.
    """

    var func: Self.FuncType

    @always_inline
    def __call__[
        width: Int, call_rank: Int, alignment: Int = 1
    ](self, indices: IndexList[call_rank]):
        comptime assert call_rank == Self.rank
        self.func[width, alignment](Coord(indices))


@always_inline
def _elementwise_impl[
    simd_width: Int,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    /,
    *,
    target: StaticString = "cpu",
    trace_description: StaticString,
](func: FuncType, shape: Coord, context: DeviceContext) raises:
    @always_inline
    @parameter
    def description_fn() -> String:
        var shape_str = trace_arg("shape", coord_to_index_list(shape))
        var vector_width_str = String(t"vector_width={simd_width}")
        return ";".join(Span([shape_str^, vector_width_str^]))

    # Intern the kind string as a static string so we don't allocate.
    comptime d = trace_description
    comptime desc = String(t"({d})") if d else ""
    comptime kind = get_static_string["elementwise", desc]()

    with Trace[TraceLevel.OP, target=target](
        kind,
        Trace[TraceLevel.OP]._get_detail_str[description_fn](),
        task_id=get_safe_task_id(context),
    ):
        # Check the host (CPU) path first: a CPU-targeted op must run on the
        # host even in an accelerator build. Only after ruling out CPU do we
        # consult the accelerator plugin, so host ops never touch `PluginForTarget`.
        # TODO(DRIV-186): GPUInfo should handle CPU device,
        # Should not need to additionally check accelerator arch here
        comptime if is_cpu[target]():

            @always_inline
            def func_wrap_cpu[
                width: Int, alignment: Int = 1
            ](coords: Coord) {imm}:
                func[width, alignment](coords)

            _elementwise_impl_cpu[
                simd_width=simd_width,
                trace_description=trace_description,
            ](func_wrap_cpu, shape=shape, ctx=Optional(context))
        elif _accelerator_arch() != "" and PluginForTarget[
            context.default_device_info.target()
        ]._handles_elementwise:
            comptime plugin = PluginForTarget[
                context.default_device_info.target()
            ]
            return plugin.elementwise_fn[shape.rank, simd_width](
                _CoordToIndexListAdapter[shape.rank, FuncType](func),
                coord_to_index_list(shape),
                context,
            )
        else:
            CompilationTarget.unsupported_target_error[
                operation=__get_current_function_name()
            ]()
