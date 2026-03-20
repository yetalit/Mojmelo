Mojo struct

# `UnionFind`

```mojo
@memory_only
struct UnionFind
```

## Fields

- **parent** (`List[Scalar[DType.int]]`)
- **rank** (`List[Scalar[DType.int]]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
def __init__(out self, size: Int)
```

**Args:**

- **size** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `find`

```mojo
def find(mut self, x: Scalar[DType.int]) -> Scalar[DType.int]
```

**Args:**

- **self** (`Self`)
- **x** (`Scalar`)

**Returns:**

`Scalar`

### `unite`

```mojo
def unite(mut self, x: Scalar[DType.int], y: Scalar[DType.int])
```

**Args:**

- **self** (`Self`)
- **x** (`Scalar`)
- **y** (`Scalar`)


