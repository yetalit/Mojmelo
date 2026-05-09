Mojo function

# `sort`

```mojo
fn sort[dtype: DType, origin: MutOrigin, //, cmp_fn: def(Scalar[dtype], Scalar[dtype]) capturing -> Bool](span: Span[Scalar[dtype], origin], indices: UnsafePointer[Int, MutAnyOrigin])
```

**Parameters:**

- **dtype** (`DType`)
- **origin** (`MutOrigin`)
- **cmp_fn** (`def(Scalar[dtype], Scalar[dtype]) capturing -> Bool`)

**Args:**

- **span** (`Span[Scalar[dtype], origin]`)
- **indices** (`UnsafePointer[Int, MutAnyOrigin]`)

