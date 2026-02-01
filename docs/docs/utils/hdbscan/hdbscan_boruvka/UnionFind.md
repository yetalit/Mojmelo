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
- **rank** (`List[Scalar[DType.index]]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, size: Int)
```

**Args:**

- **size** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `find`

```mojo
fn find(mut self, x: Scalar[DType.index]) -> Scalar[DType.index]
```

**Args:**

- **self** (`Self`)
- **x** (`Scalar`)

**Returns:**

`Scalar`

### `unite`

```mojo
fn unite(mut self, x: Scalar[DType.index], y: Scalar[DType.index])
```

**Args:**

- **self** (`Self`)
- **x** (`Scalar`)
- **y** (`Scalar`)


