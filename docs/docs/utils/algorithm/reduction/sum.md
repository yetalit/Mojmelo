Mojo function

# `sum`

```mojo
fn def sum[dtype: DType](src: Span[Scalar[dtype]]) -> Scalar[dtype]
```

Computes the sum of buffer elements.

**Parameters:**

- **dtype** (`DType`): The dtype of the input.

**Args:**

- **src** (`Span[Scalar[dtype]]`): The buffer.

**Returns:**

`Scalar[dtype]`: The sum of the buffer elements.

**Raises:**

If the operation fails.

```mojo
fn def sum[dtype: DType, InputFn1DType: def[dtype_: DType, width: Int](idx: Int) -> SIMD[dtype_, width] & RegisterPassable & ImplicitlyCopyable](length: Int, input_fn_1d: InputFn1DType) -> Scalar[dtype]
```

Computes the sum of a 1D array using a provided input function.

This function performs a reduction (sum) over a 1-dimensional array of the specified length and data type.
The input values are provided by the `input_fn_1d` function, which takes an index and returns a SIMD vector
of the specified width and data type. The reduction is performed using a single thread for deterministic results.

**Parameters:**

- **dtype** (`DType`): The data type of the elements to sum.
- **InputFn1DType** (`def[dtype_: DType, width: Int](idx: Int) -> SIMD[dtype_, width] & RegisterPassable & ImplicitlyCopyable`): The type of `input_fn_1d` (inferred).

**Args:**

- **length** (`Int`): The number of elements in the 1D array.
- **input_fn_1d** (`InputFn1DType`): A function that takes a data type, SIMD width, and
  index, and returns a SIMD vector of input values.

**Returns:**

`Scalar[dtype]`: The sum of all elements as a scalar of the specified data type.

**Raises:**

Any exception raised by the input function or reduction process.

