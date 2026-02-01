Mojo function

# `sort`

```mojo
fn sort[T: Copyable & Movable, origin: MutOrigin, //, cmp_fn: fn(T, T) capturing -> Bool, *, __disambiguate: NoneType = None](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index], MutAnyOrigin])
```

**Parameters:**

- **T** (`Copyable & Movable`)
- **origin** (`MutOrigin`)
- **cmp_fn** (`fn(T, T) capturing -> Bool`)
- **__disambiguate** (`NoneType`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[dtype: DType, origin: MutOrigin, //, cmp_fn: fn(Scalar[dtype], Scalar[dtype]) capturing -> Bool](span: Span[Scalar[dtype], origin], indices: UnsafePointer[Scalar[DType.index], MutAnyOrigin])
```

**Parameters:**

- **dtype** (`DType`)
- **origin** (`MutOrigin`)
- **cmp_fn** (`fn(Scalar[dtype], Scalar[dtype]) capturing -> Bool`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[origin: MutOrigin, //, cmp_fn: fn(Int, Int) capturing -> Bool](span: Span[Int, origin], indices: UnsafePointer[Scalar[DType.index], MutAnyOrigin])
```

**Parameters:**

- **origin** (`MutOrigin`)
- **cmp_fn** (`fn(Int, Int) capturing -> Bool`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[origin: MutOrigin, //](span: Span[Int, origin], indices: UnsafePointer[Scalar[DType.index], MutAnyOrigin])
```

**Parameters:**

- **origin** (`MutOrigin`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[T: Copyable & Movable & Comparable, origin: MutOrigin, //](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index], MutAnyOrigin])
```

**Parameters:**

- **T** (`Copyable & Movable & Comparable`)
- **origin** (`MutOrigin`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

