Mojo struct

# `KDTreeNode`

```mojo
@memory_only
struct KDTreeNode
```

## Fields

- **cut_dim** (`Int`)
- **cut_val** (`SIMD[float32, 1]`)
- **cut_val_left** (`SIMD[float32, 1]`)
- **cut_val_right** (`SIMD[float32, 1]`)
- **l** (`Int`)
- **u** (`Int`)
- **box** (`List[interval]`)
- **left** (`UnsafePointer[KDTreeNode]`)
- **right** (`UnsafePointer[KDTreeNode]`)
- **metric** (`fn(SIMD[float32, 1]) -> SIMD[float32, 1]`)

## Implemented traits

`AnyType`, `Copyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, dim: Int, metric: fn(SIMD[float32, 1]) -> SIMD[float32, 1])
```

**Args:**

- **dim** (`Int`)
- **metric** (`fn(SIMD[float32, 1]) -> SIMD[float32, 1]`)
- **self** (`Self`)

**Returns:**

`Self`

### `search`

```mojo
fn search(self, mut sr: SearchRecord)
```

**Args:**

- **self** (`Self`)
- **sr** (`SearchRecord`)

### `box_in_search_range`

```mojo
fn box_in_search_range(self, sr: SearchRecord) -> Bool
```

**Args:**

- **self** (`Self`)
- **sr** (`SearchRecord`)

**Returns:**

`Bool`

### `process_terminal_node`

```mojo
fn process_terminal_node(self, mut sr: SearchRecord)
```

**Args:**

- **self** (`Self`)
- **sr** (`SearchRecord`)

### `process_terminal_node_fixedball`

```mojo
fn process_terminal_node_fixedball(self, sr: SearchRecord)
```

**Args:**

- **self** (`Self`)
- **sr** (`SearchRecord`)


