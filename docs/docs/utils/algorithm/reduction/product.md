Mojo function

# `product`

```mojo
fn def product[dtype: DType, input_fn: def[width: Int, rank: Int](IndexList[rank]) capturing thin -> SIMD[dtype, width], output_fn: def[width: SIMDLength, rank: Int](IndexList[rank], SIMD[dtype, width]) capturing thin -> None, /, target: StringSpan[ImmStaticOrigin] = StringSpan("cpu"), *, reduce_dim: Int](input_shape: Coord)
```

Computes the product across the input and output shape. This performs the product computation on the domain specified by `input_shape`, loading the inputs using the `input_fn`. The results are stored using the `output_fn`.

**Parameters:**

- **dtype** (`DType`): The dtype of the input and output.
- **input_fn** (`def[width: Int, rank: Int](IndexList[rank]) capturing thin -> SIMD[dtype, width]`): The function to load the input.
- **output_fn** (`def[width: SIMDLength, rank: Int](IndexList[rank], SIMD[dtype, width]) capturing thin -> None`): The function to store the output.
- **target** (`StringSpan[ImmStaticOrigin]`): The target to run on.
- **reduce_dim** (`Int`): The axis to perform the product on.

**Args:**

- **input_shape** (`Coord`): The input shape.

**Raises:**

If the operation fails.

```mojo
fn def product[dtype: DType](src: Span[Scalar[dtype]]) -> Scalar[dtype]
```

Computes the product of the buffer elements.

**Parameters:**

- **dtype** (`DType`): The dtype of the input.

**Args:**

- **src** (`Span[Scalar[dtype]]`): The buffer.

**Returns:**

`Scalar[dtype]`: The product of the buffer elements.

**Raises:**

If the operation fails.

