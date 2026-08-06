Mojo function

# `cumsum`

```mojo
fn def cumsum[dtype: DType](dst: Span[Scalar[dtype]], src: Span[Scalar[dtype]])
```

Computes the cumulative sum of all elements in a buffer.    dst[i] = src[i] + src[i-1] + ... + src[0].

**Parameters:**

- **dtype** (`DType`): The dtype of the input.

**Args:**

- **dst** (`Span[Scalar[dtype]]`): The buffer that stores the result of cumulative sum operation.
- **src** (`Span[Scalar[dtype]]`): The buffer of elements for which the cumulative sum is computed.

