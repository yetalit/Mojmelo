Mojo struct

# `UnionFind`

```mojo
@memory_only
struct UnionFind
```

## Fields

- **parent** (`List[Scalar[DType.int]]`)
- **size** (`List[Scalar[DType.int]]`)
- **next_label** (`Scalar[DType.int]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
def __init__(out self, N: Int)
```

**Args:**

- **N** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `union`

```mojo
def union(mut self, m: Scalar[DType.int], n: Scalar[DType.int])
```

**Args:**

- **self** (`Self`)
- **m** (`Scalar`)
- **n** (`Scalar`)

### `fast_find`

```mojo
def fast_find(mut self, var n: Scalar[DType.int]) -> Scalar[DType.int]
```

**Args:**

- **self** (`Self`)
- **n** (`Scalar`)

**Returns:**

`Scalar`


