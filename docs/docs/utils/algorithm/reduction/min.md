Mojo function

# `min`

```mojo
fn def min[dtype: DType](src: Span[Scalar[dtype]]) -> Scalar[dtype]
```

Computes the min element in a buffer.

**Parameters:**

- **dtype** (`DType`): The dtype of the input.

**Args:**

- **src** (`Span[Scalar[dtype]]`): The buffer.

**Returns:**

`Scalar[dtype]`: The minimum of the buffer elements.

**Raises:**

If the operation fails.

