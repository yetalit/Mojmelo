Mojo function

# `reduce`

```mojo
fn def reduce[reduce_fn: def[acc_type: DType, dtype: DType, width: SIMDLength](SIMD[acc_type, width], SIMD[dtype, width]) capturing thin -> SIMD[acc_type, width], dtype: DType](src: Span[Scalar[dtype]], init: Scalar[dtype]) -> Scalar[dtype]
```

Computes a custom reduction of buffer elements.

**Parameters:**

- **reduce_fn** (`def[acc_type: DType, dtype: DType, width: SIMDLength](SIMD[acc_type, width], SIMD[dtype, width]) capturing thin -> SIMD[acc_type, width]`): The lambda implementing the reduction.
- **dtype** (`DType`): The dtype of the input.

**Args:**

- **src** (`Span[Scalar[dtype]]`): The input buffer.
- **init** (`Scalar[dtype]`): The initial value to use in accumulator.

**Returns:**

`Scalar[dtype]`: The computed reduction value.

**Raises:**

If the operation fails.

