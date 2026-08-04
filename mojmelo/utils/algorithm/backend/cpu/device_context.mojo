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

from std.builtin.coroutine import (
    AnyCoroutine,
    _coro_resume_fn,
    _coro_destroy_fn,
)
from std.collections.optional import OptionalReg
from std.reflection import call_location, SourceLocation
from std.ffi import (
    external_call,
    c_char,
    CStringSlice,
    _CPointer,
)

comptime _CString[origin: ImmOrigin = ImmUntrackedOrigin] = Optional[
    CStringSlice[origin]
]
comptime _DeviceContextPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = _CPointer[_DeviceContextCpp, origin]

def _string_from_owned_charptr(c_str: _CString) -> String:
    var result = String()
    if c_str:
        result = String(unsafe_from_utf8_ptr=c_str.unsafe_value().unsafe_ptr())
    # void AsyncRT_DeviceContext_strfree(const char* ptr)
    external_call["AsyncRT_DeviceContext_strfree", NoneType](c_str)
    return result^

@no_inline
def _raise_checked_impl(
    err_msg: _CString, msg: String, location: SourceLocation
) raises:
    var err = _string_from_owned_charptr(err_msg)
    raise Error(location.prefix(err + ((" " + msg) if msg else "")))

@always_inline
def _checked(
    err: _CString,
    *,
    msg: String = "",
    location: OptionalReg[SourceLocation] = None,
) raises:
    if err:
        _raise_checked_impl(err, msg, location.or_else(call_location()))

# Create empty structs to ensure dtype checking when using the C++ handles.
struct _DeviceContextCpp:
    pass

struct DeviceContext(ImplicitlyCopyable, RegisterPassable):

    var _handle: _DeviceContextPtr[mut=True]
    var _owning: Bool

    @always_inline
    def __init__(
        out self,
        device_id: Int = 0,
        *,
        var api: String = "cpu",
    ) raises:
        # const char *AsyncRT_DeviceContext_create(const DeviceContext **result, const char *api, int id)
        var result: _DeviceContextPtr[mut=True] = {}
        _checked(
            external_call[
                "AsyncRT_DeviceContext_create",
                _CString[],
                Pointer[_DeviceContextPtr[mut=True], origin_of(result)],
                Pointer[c_char, ImmutAnyOrigin],
                Int32,
            ](
                Pointer(to=result),
                api.as_c_string_slice().unsafe_ptr().as_unsafe_any_origin(),
                Int32(device_id),
            )
        )
        self._handle = result
        self._owning = True

    @always_inline
    def enqueue_cpu_range[
        FuncType: def(Int) -> None,
    ](self, func: FuncType, count: Int) raises:
        """Enqueues a function to be executed in parallel over a 1D range.

        The function is called as `func(i)` for each `i` in `range(count)`.

        Instances of the function are executed in parallel, but it is not
        guaranteed that all instances will execute simultaneously.

        Parameters:
            FuncType: The type of function to execute.

        Args:
            func: The function closure to execute.
            count: The number of parallel instances of the function to enqueue.

        Raises:
            If the operation fails.
        """
        var handles = List[AnyCoroutine](capacity=count)

        async def wrapper(idx: Int) capturing -> None:
            func(idx)

        for j in range(count):
            var coro = wrapper(j)
            coro._set_noop_callback()
            handles.append(coro^._take_handle())

        _checked(
            external_call[
                "AsyncRT_DeviceContext_enqueueHostFunctionRange",
                _CString[],
            ](
                self._handle,
                _coro_resume_fn,
                _coro_destroy_fn,
                handles.unsafe_ptr(),
                count,
            )
        )

    @always_inline
    def synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.


        Raises:
            If the operation fails. This should never be necessary when
            writing a custom operation.
        """
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_synchronize",
                _CString[],
                _DeviceContextPtr[mut=True],
            ](
                self._handle,
            ),
            location=call_location(),
        )

    def __deinit__(deinit self):
        """Releases resources associated with this device context.

        This destructor decrements the reference count of the native device context.
        When the reference count reaches zero, the underlying resources are released,
        including any cached memory buffers and compiled device functions.
        """
        if not self._owning:
            return
        # Decrement the reference count held by this struct.
        #
        # void AsyncRT_DeviceContext_release(const DeviceContext *ctx)
        external_call[
            "AsyncRT_DeviceContext_release",
            NoneType,
            _DeviceContextPtr[mut=True],
        ](self._handle)
