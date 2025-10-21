Mojo struct

# `svm_problem`

```mojo
@memory_only
struct svm_problem
```

## Aliases

- `__del__is_trivial = True`

## Fields

- **l** (`Int`)
- **y** (`UnsafePointer[Float64]`)
- **x** (`UnsafePointer[UnsafePointer[svm_node]]`)

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


