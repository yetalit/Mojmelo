Mojo struct

# `head_t`

```mojo
@memory_only
struct head_t
```

## Fields

- **prev** (`UnsafePointer[head_t]`)
- **next** (`UnsafePointer[head_t]`)
- **data** (`UnsafePointer[SIMD[float32, 1]]`)

## Implemented traits

`AnyType`, `Copyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, p: UnsafePointer[head_t], n: UnsafePointer[head_t], _len: Int)
```

**Args:**

- **p** (`UnsafePointer`)
- **n** (`UnsafePointer`)
- **_len** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn __del__(var self)
```

**Args:**

- **self** (`Self`)


