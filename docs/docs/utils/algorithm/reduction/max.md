Mojo function

# `max`

```mojo
fn def max[dtype: DType](src: Span[Scalar[dtype]]) -> Scalar[dtype]
```

Computes the max element in a buffer.

**Parameters:**

- **dtype** (`DType`): The dtype of the input.

**Args:**

- **src** (`Span[Scalar[dtype]]`): The buffer.

**Returns:**

`Scalar[dtype]`: The maximum of the buffer elements.

**Raises:**

If the operation fails.

