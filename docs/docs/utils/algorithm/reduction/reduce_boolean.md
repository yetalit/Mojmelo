Mojo function

# `reduce_boolean`

```mojo
fn def reduce_boolean[reduce_fn: def[dtype: DType, width: SIMDLength](SIMD[dtype, width]) capturing thin -> Bool, continue_fn: def(Bool) capturing thin -> Bool, dtype: DType](src: Span[Scalar[dtype]], init: Bool) -> Bool
```

Computes a bool reduction of buffer elements. The reduction will early exit if the `continue_fn` returns False.

**Parameters:**

- **reduce_fn** (`def[dtype: DType, width: SIMDLength](SIMD[dtype, width]) capturing thin -> Bool`): A boolean reduction function. This function is used to reduce
  a vector to a scalar. E.g. when we got `8xfloat32` vector and want to
  reduce it to a `bool`.
- **continue_fn** (`def(Bool) capturing thin -> Bool`): A function to indicate whether we want to continue
  processing the rest of the iterations. This takes the result of the
  reduce_fn and returns True to continue processing and False to early
  exit.
- **dtype** (`DType`): The dtype of the input.

**Args:**

- **src** (`Span[Scalar[dtype]]`): The input buffer.
- **init** (`Bool`): The initial value to use.

**Returns:**

`Bool`: The computed reduction value.

