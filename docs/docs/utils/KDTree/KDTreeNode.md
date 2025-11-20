Mojo struct

# `KDTreeNode`

```mojo
@memory_only
struct KDTreeNode
```

## Aliases

- `__del__is_trivial = False`
- `__moveinit__is_trivial = True`
- `__copyinit__is_trivial = False`

## Fields

- **cut_dim** (`Int`)
- **cut_val** (`Float32`)
- **cut_val_left** (`Float32`)
- **cut_val_right** (`Float32`)
- **l** (`Int`)
- **u** (`Int`)
- **box** (`List[interval]`)
- **left** (`UnsafePointer[KDTreeNode, MutAnyOrigin]`)
- **right** (`UnsafePointer[KDTreeNode, MutAnyOrigin]`)
- **metric** (`fn(Float32) -> Float32`)

## Implemented traits

`AnyType`, `Copyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, dim: Int, metric: fn(Float32) -> Float32)
```

**Args:**

- **dim** (`Int`)
- **metric** (`fn(Float32) -> Float32`)
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


