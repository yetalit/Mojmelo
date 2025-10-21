Mojo struct

# `head_t`

```mojo
@memory_only
struct head_t
```

## Aliases

- `__del__is_trivial = True`

## Fields

- **prev** (`UnsafePointer[head_t]`)
- **next** (`UnsafePointer[head_t]`)
- **data** (`UnsafePointer[Float32]`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self)
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`


