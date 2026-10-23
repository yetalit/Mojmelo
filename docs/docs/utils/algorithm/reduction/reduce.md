Mojo function

# `reduce`

```mojo
fn def reduce[dtype: DType, ReduceFnType: def[acc_type: DType, dtype: DType, width: SIMDLength](SIMD[acc_type, width], SIMD[dtype, width]) -> SIMD[acc_type, width] & RegisterPassable & ImplicitlyCopyable, //](src: Span[Scalar[dtype]], init: Scalar[dtype], reduce_fn: ReduceFnType) -> Scalar[dtype]
```

Computes a custom reduction of buffer elements.

**Parameters:**

- **dtype** (`DType`): The dtype of the input.
- **ReduceFnType** (`def[acc_type: DType, dtype: DType, width: SIMDLength](SIMD[acc_type, width], SIMD[dtype, width]) -> SIMD[acc_type, width] & RegisterPassable & ImplicitlyCopyable`): The type of `reduce_fn` (inferred).

**Args:**

- **src** (`Span[Scalar[dtype]]`): The input buffer.
- **init** (`Scalar[dtype]`): The initial value to use in accumulator.
- **reduce_fn** (`ReduceFnType`): The lambda implementing the reduction.

**Returns:**

`Scalar[dtype]`: The computed reduction value.

**Raises:**

If the operation fails.

