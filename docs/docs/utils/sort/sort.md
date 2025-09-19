Mojo function

# `sort`

```mojo
fn sort[: origin.set, T: Copyable & Movable, origin: MutableOrigin, //, cmp_fn: fn(T, T) capturing -> Bool](span: Span[T, origin], indices: UnsafePointer[SIMD[index, 1]])
```

**Parameters:**

- **** (`origin.set`)
- **T** (`Copyable & Movable`)
- **origin** (`MutableOrigin`)
- **cmp_fn** (`fn(T, T) capturing -> Bool`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

```mojo
fn sort[: origin.set, origin: MutableOrigin, //, cmp_fn: fn(Int, Int) capturing -> Bool](span: Span[Int, origin], indices: UnsafePointer[SIMD[index, 1]])
```

**Parameters:**

- **** (`origin.set`)
- **origin** (`MutableOrigin`)
- **cmp_fn** (`fn(Int, Int) capturing -> Bool`)

**Args:**

- **span** (`Span`)
- **indices** (`UnsafePointer`)

