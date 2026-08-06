Mojo function

# `sum`

```mojo
fn def sum[dtype: DType, input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing thin -> SIMD[dtype, width], output_fn: def[width: SIMDLength, rank: Int](IndexList[rank], SIMD[dtype, width]) capturing thin -> None, /, target: StringSpan[ImmStaticOrigin] = StringSpan("cpu"), *, reduce_dim: Int](input_shape: Coord)
```

Computes the sum across the input and output shape.

This performs the sum computation on the domain specified by `input_shape`,
loading the inputs using the `input_fn`. The results are stored using
the `output_fn`.

**Parameters:**

- **dtype** (`DType`): The dtype of the input and output.
- **input_fn** (`def[width: Int, rank: Int](IndexList[rank]) capturing thin -> SIMD[dtype, width]`): The function to load the input.
- **output_fn** (`def[width: SIMDLength, rank: Int](IndexList[rank], SIMD[dtype, width]) capturing thin -> None`): The function to store the output.
- **target** (`StringSpan[ImmStaticOrigin]`): The target to run on.
- **reduce_dim** (`Int`): The axis to perform the sum on.

**Args:**

- **input_shape** (`Coord`): The input shape.

**Raises:**

If the operation fails.

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
fn def sum[dtype: DType, input_fn_1d: def[dtype_: DType, width: Int](idx: Int) capturing thin -> SIMD[dtype_, width]](length: Int) -> Scalar[dtype]
```

Computes the sum of a 1D array using a provided input function.

This function performs a reduction (sum) over a 1-dimensional array of the specified length and data type.
The input values are provided by the `input_fn_1d` function, which takes an index and returns a SIMD vector
of the specified width and data type. The reduction is performed using a single thread for deterministic results.

**Parameters:**

- **dtype** (`DType`): The data type of the elements to sum.
- **input_fn_1d** (`def[dtype_: DType, width: Int](idx: Int) capturing thin -> SIMD[dtype_, width]`): A function that takes a data type, SIMD width, and index, and returns a SIMD vector of input values.

**Args:**

- **length** (`Int`): The number of elements in the 1D array.

**Returns:**

`Scalar[dtype]`: The sum of all elements as a scalar of the specified data type.

**Raises:**

Any exception raised by the input function or reduction process.

