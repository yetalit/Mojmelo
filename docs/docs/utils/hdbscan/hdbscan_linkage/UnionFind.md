Mojo struct

# `UnionFind`

```mojo
@memory_only
struct UnionFind
```

## Aliases

- `__del__is_trivial = False`

## Fields

- **parent** (`List[Scalar[DType.index]]`)
- **size** (`List[Scalar[DType.index]]`)
- **next_label** (`Int`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, N: Int)
```

**Args:**

- **N** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `union`

```mojo
fn union(mut self, m: Scalar[DType.index], n: Scalar[DType.index])
```

**Args:**

- **self** (`Self`)
- **m** (`Scalar`)
- **n** (`Scalar`)

### `fast_find`

```mojo
fn fast_find(mut self, var n: Scalar[DType.index]) -> Scalar[DType.index]
```

**Args:**

- **self** (`Self`)
- **n** (`Scalar`)

**Returns:**

`Scalar`


