Mojo function

# `product`

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

