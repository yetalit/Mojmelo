Mojo struct

# `UnionFind`

```mojo
@memory_only
struct UnionFind
```

## Fields

- **parent** (`List[Int]`)
- **size** (`List[Int]`)
- **next_label** (`Int`)

## Implemented traits

`AnyType`, `ImplicitlyDeletable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, N: Int)
```

**Args:**

- **N** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `union`

```mojo
fn def union(mut self, m: Int, n: Int)
```

**Args:**

- **self** (`Self`)
- **m** (`Int`)
- **n** (`Int`)

### `fast_find`

```mojo
fn def fast_find(mut self, var n: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **n** (`Int`)

**Returns:**

`Int`


