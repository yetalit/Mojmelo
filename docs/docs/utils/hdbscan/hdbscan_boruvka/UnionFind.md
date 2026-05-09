Mojo struct

# `UnionFind`

```mojo
@memory_only
struct UnionFind
```

## Fields

- **parent** (`List[Int]`)
- **rank** (`List[Int]`)

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
fn find(mut self, x: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **x** (`Int`)

**Returns:**

`Int`

### `unite`

```mojo
fn unite(mut self, x: Int, y: Int)
```

**Args:**

- **self** (`Self`)
- **x** (`Int`)
- **y** (`Int`)


