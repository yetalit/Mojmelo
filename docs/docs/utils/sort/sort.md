Mojo function

# `sort`

```mojo
fn def sort[dtype: DType, origin: MutOrigin, //, cmp_fn: def(Scalar[dtype], Scalar[dtype]) capturing thin -> Bool](span: Span[Scalar[dtype], origin], indices: Pointer[Int, MutUntrackedOrigin])
```

**Parameters:**

- **dtype** (`DType`)
- **origin** (`MutOrigin`)
- **cmp_fn** (`def(Scalar[dtype], Scalar[dtype]) capturing thin -> Bool`)

**Args:**

- **span** (`Span[Scalar[dtype], origin]`)
- **indices** (`Pointer[Int, MutUntrackedOrigin]`)

