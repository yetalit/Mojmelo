from sys import bitwidthof
from bit import count_leading_zeros
from memory import UnsafePointer
from math import ceil

@always_inline
fn _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    var log2 = Int(
        (bitwidthof[DType.index]() - 1) ^ count_leading_zeros(size | 1)
    )
    # The number 1.3 was chosen by experimenting the max stack size for random
    # input. This also depends on insertion_sort_threshold
    return max(2, Int(ceil(1.3 * log2)))

@fieldwise_init("implicit")
struct _SortWrapper[T: Copyable & Movable](Copyable, Movable):
    var data: T

    fn __init__(out self, *, other: Self):
        self.data = other.data

# ===----------------------------------------------------------------------===#
# sort
# ===-----------------------------------------------------------------------===#

alias insertion_sort_threshold = 32

@always_inline
fn _insertion_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]]):
    """Sort the array[start:end] slice"""
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)

    for i in range(1, size):
        var value = array[i]
        var value_b = indices[i]
        var j = i

        # Find the placement of the value in the array, shifting as we try to
        # find the position. Throughout, we assume array[start:i] has already
        # been sorted.
        while j > 0 and cmp_fn(value, array[j - 1]):
            array[j] = array[j - 1]
            indices[j] = indices[j - 1]
            j -= 1

        array[j] = value
        indices[j] = value_b


# put everything thats "<" to the left of pivot
@always_inline
fn _quicksort_partition_right[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]]) -> Int:
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)

    var left = 1
    var right = size - 1
    var pivot_value = array[0]

    while True:
        # no need for left < right since quick sort pick median of 3 as pivot
        while cmp_fn(array[left], pivot_value):
            left += 1
        while left < right and not cmp_fn(array[right], pivot_value):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            swap(array[pivot_pos], array[0])
            swap(indices[pivot_pos], indices[0])
            return pivot_pos
        swap(array[left], array[right])
        swap(indices[left], indices[right])
        left += 1
        right -= 1


# put everything thats "<=" to the left of pivot
@always_inline
fn _quicksort_partition_left[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]]) -> Int:
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)

    var left = 1
    var right = size - 1
    var pivot_value = array[0]

    while True:
        while left < right and not cmp_fn(pivot_value, array[left]):
            left += 1
        while cmp_fn(pivot_value, array[right]):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            swap(array[pivot_pos], array[0])
            swap(indices[pivot_pos], indices[0])
            return pivot_pos
        swap(array[left], array[right])
        swap(indices[left], indices[right])
        left += 1
        right -= 1


@always_inline
fn _delegate_small_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]]):
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)
    if size == 2:
        _small_sort[2, T, cmp_fn](array, indices)

        return
    if size == 3:
        _small_sort[3, T, cmp_fn](array, indices)
        return

    if size == 4:
        _small_sort[4, T, cmp_fn](array, indices)
        return

    if size == 5:
        _small_sort[5, T, cmp_fn](array, indices)
        return


# FIXME (MSTDL-808): Using _Pair over Span results in 1-3% improvement
# struct _Pair[T: AnyType]:
#     var ptr: UnsafePointer[T]
#     var len: Int


@always_inline
fn _quicksort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]]):
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)
    if size == 0:
        return

    # Work with an immutable span so we don't run into exclusivity problems with
    # the List[Span].
    alias ImmSpan = span.Immutable

    var stack = List[ImmSpan](capacity=_estimate_initial_height(size))
    var stack_b = List[UnsafePointer[Scalar[DType.index]]](capacity=stack.capacity)
    stack.append(span)
    stack_b.append(indices)
    while len(stack) > 0:
        var imm_interval = stack.pop()
        var interval_b = stack_b.pop()
        var imm_ptr = imm_interval.unsafe_ptr()
        var mut_ptr = imm_ptr.origin_cast[mut=True, origin=MutableAnyOrigin]()
        var len = len(imm_interval)
        var interval = Span[T, MutableAnyOrigin](ptr=mut_ptr, length=len)

        if len <= 5:
            _delegate_small_sort[cmp_fn](interval, interval_b)
            continue

        if len < insertion_sort_threshold:
            _insertion_sort[cmp_fn](interval, interval_b)
            continue

        # pick median of 3 as pivot
        _sort3[T, cmp_fn](mut_ptr, len >> 1, 0, len - 1, interval_b)

        # if ptr[-1] == pivot_value, then everything in between will
        # be the same, so no need to recurse that interval
        # already have array[-1] <= array[0]
        if mut_ptr > array and not cmp_fn(imm_ptr[-1], imm_ptr[0]):
            var pivot = _quicksort_partition_left[cmp_fn](interval, interval_b)
            if len > pivot + 2:
                stack.append(
                    ImmSpan(ptr=imm_ptr + pivot + 1, length=len - pivot - 1)
                )
                stack_b.append(
                    interval_b + pivot + 1
                )
            continue

        var pivot = _quicksort_partition_right[cmp_fn](interval, interval_b)

        if len > pivot + 2:
            stack.append(
                ImmSpan(ptr=imm_ptr + pivot + 1, length=len - pivot - 1)
            )
            stack_b.append(
                interval_b + pivot + 1
            )

        if pivot > 1:
            stack.append(ImmSpan(ptr=imm_ptr, length=pivot))
            stack_b.append(interval_b)


# Junction from public to private API
fn _sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool
](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]]):
    if len(span) <= 5:
        _delegate_small_sort[cmp_fn](span, indices)
        return

    if len(span) < insertion_sort_threshold:
        _insertion_sort[cmp_fn](span, indices)
        return

    _quicksort[cmp_fn](span, indices)

# TODO (MSTDL-766): The Int and Scalar[T] overload should be remove
# (same for partition)
fn sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool
](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]]):

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[T], rhs: _SortWrapper[T]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _sort[_cmp_fn](span, indices)


fn sort[
    origin: MutableOrigin, //,
    cmp_fn: fn (Int, Int) capturing [_] -> Bool
](span: Span[Int, origin], indices: UnsafePointer[Scalar[DType.index]]):

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[Int], rhs: _SortWrapper[Int]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _sort[_cmp_fn](span, indices)

# ===-----------------------------------------------------------------------===#
# sort networks
# ===-----------------------------------------------------------------------===#

@always_inline
fn _sort2[
    T: Copyable & Movable,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](
    array: UnsafePointer[
        T, address_space = AddressSpace.GENERIC, mut=True, **_
    ],
    offset0: Int,
    offset1: Int,
    indices: UnsafePointer[Scalar[DType.index]]
):
    var a = array[offset0]
    var b = array[offset1]
    if not cmp_fn(a, b):
        array[offset0] = b
        array[offset1] = a
        swap(indices[offset0], indices[offset1])

@always_inline
fn _sort3[
    T: Copyable & Movable,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](
    array: UnsafePointer[
        T, address_space = AddressSpace.GENERIC, mut=True, **_
    ],
    offset0: Int,
    offset1: Int,
    offset2: Int,
    indices: UnsafePointer[Scalar[DType.index]]
):
    _sort2[T, cmp_fn](array, offset0, offset1, indices)
    _sort2[T, cmp_fn](array, offset1, offset2, indices)
    _sort2[T, cmp_fn](array, offset0, offset1, indices)


@always_inline
fn _sort_partial_3[
    T: Copyable & Movable,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](
    array: UnsafePointer[
        T, address_space = AddressSpace.GENERIC, mut=True, **_
    ],
    offset0: Int,
    offset1: Int,
    offset2: Int,
    indices: UnsafePointer[Scalar[DType.index]]
):
    var a = array[offset0]
    var b = array[offset1]
    var c = array[offset2]
    var ab = indices[offset0]
    var bb = indices[offset1]
    var cb = indices[offset2]
    var r = cmp_fn(c, a)
    var t = c if r else a
    var tb = cb if r else ab
    if r:
        array[offset2] = a
        indices[offset2] = ab
    if cmp_fn(b, t):
        array[offset0] = b
        array[offset1] = t
        indices[offset0] = bb
        indices[offset1] = tb
    elif r:
        array[offset0] = t
        indices[offset0] = tb


@always_inline
fn _small_sort[
    n: Int,
    T: Copyable & Movable,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](array: UnsafePointer[T, address_space = AddressSpace.GENERIC, mut=True, **_], indices: UnsafePointer[Scalar[DType.index]]):
    @parameter
    if n == 2:
        _sort2[T, cmp_fn](array, 0, 1, indices)
        return

    @parameter
    if n == 3:
        _sort2[T, cmp_fn](array, 1, 2, indices)
        _sort_partial_3[T, cmp_fn](array, 0, 1, 2, indices)
        return

    @parameter
    if n == 4:
        _sort2[T, cmp_fn](array, 0, 2, indices)
        _sort2[T, cmp_fn](array, 1, 3, indices)
        _sort2[T, cmp_fn](array, 0, 1, indices)
        _sort2[T, cmp_fn](array, 2, 3, indices)
        _sort2[T, cmp_fn](array, 1, 2, indices)
        return

    @parameter
    if n == 5:
        _sort2[T, cmp_fn](array, 0, 1, indices)
        _sort2[T, cmp_fn](array, 3, 4, indices)
        _sort_partial_3[T, cmp_fn](array, 2, 3, 4, indices)
        _sort2[T, cmp_fn](array, 1, 4, indices)
        _sort_partial_3[T, cmp_fn](array, 0, 2, 3, indices)
        _sort_partial_3[T, cmp_fn](array, 1, 2, 3, indices)
        return
