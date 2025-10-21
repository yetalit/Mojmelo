Mojo function

# `sort`

```mojo
fn sort[T: Copyable & Movable, origin: MutableOrigin, //, cmp_fn: fn(T, T) capturing -> Bool, *, __disambiguate: NoneType = None](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]])
```

**Parameters:**

- **T** (`Copyable & Movable`)
- **origin** (`MutableOrigin`)
- **cmp_fn** (`fn(T, T) capturing -> Bool`)
- **__disambiguate** (`NoneType`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[dtype: DType, origin: MutableOrigin, //, cmp_fn: fn(Scalar[dtype], Scalar[dtype]) capturing -> Bool](span: Span[Scalar[dtype], origin], indices: UnsafePointer[Scalar[DType.index]])
```

**Parameters:**

- **dtype** (`DType`)
- **origin** (`MutableOrigin`)
- **cmp_fn** (`fn(Scalar[dtype], Scalar[dtype]) capturing -> Bool`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[origin: MutableOrigin, //, cmp_fn: fn(Int, Int) capturing -> Bool](span: Span[Int, origin], indices: UnsafePointer[Scalar[DType.index]])
```

**Parameters:**

- **origin** (`MutableOrigin`)
- **cmp_fn** (`fn(Int, Int) capturing -> Bool`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[origin: MutableOrigin, //](span: Span[Int, origin], indices: UnsafePointer[Scalar[DType.index]])
```

**Parameters:**

- **origin** (`MutableOrigin`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[T: Copyable & Movable & EqualityComparable & LessThanComparable & GreaterThanComparable & LessThanOrEqualComparable & GreaterThanOrEqualComparable, origin: MutableOrigin, //](span: Span[T, origin], indices: UnsafePointer[Scalar[DType.index]])
```

**Parameters:**

- **T** (`Copyable & Movable & EqualityComparable & LessThanComparable & GreaterThanComparable & LessThanOrEqualComparable & GreaterThanOrEqualComparable`)
- **origin** (`MutableOrigin`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

