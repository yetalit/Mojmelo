Mojo function

# `sort`

```mojo
def sort[T: Copyable & Movable, origin: MutOrigin, //, cmp_fn: def(T, T) capturing -> Bool, *, __disambiguate: NoneType = None](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.int], MutAnyOrigin])
```

**Parameters:**

- **T** (`Copyable & Movable`)
- **origin** (`MutOrigin`)
- **cmp_fn** (`def(T, T) capturing -> Bool`)
- **__disambiguate** (`NoneType`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
def sort[dtype: DType, origin: MutOrigin, //, cmp_fn: def(Scalar[dtype], Scalar[dtype]) capturing -> Bool](span: Span[Scalar[dtype], origin], indices: UnsafePointer[Scalar[DType.int], MutAnyOrigin])
```

**Parameters:**

- **dtype** (`DType`)
- **origin** (`MutOrigin`)
- **cmp_fn** (`def(Scalar[dtype], Scalar[dtype]) capturing -> Bool`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
def sort[origin: MutOrigin, //, cmp_fn: def(Int, Int) capturing -> Bool](span: Span[Int, origin], indices: UnsafePointer[Scalar[DType.int], MutAnyOrigin])
```

**Parameters:**

- **origin** (`MutOrigin`)
- **cmp_fn** (`def(Int, Int) capturing -> Bool`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
def sort[origin: MutOrigin, //](span: Span[Int, origin], indices: UnsafePointer[Scalar[DType.int], MutAnyOrigin])
```

**Parameters:**

- **origin** (`MutOrigin`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
def sort[T: Copyable & Movable & Comparable, origin: MutOrigin, //](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.int], MutAnyOrigin])
```

**Parameters:**

- **T** (`Copyable & Movable & Comparable`)
- **origin** (`MutOrigin`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

