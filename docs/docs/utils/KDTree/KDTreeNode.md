Mojo struct

# `KDTreeNode`

```mojo
@memory_only
struct KDTreeNode
```

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
- **metric** (`def(Float32) -> Float32`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
def __init__(out self, dim: Int, metric: def(Float32) -> Float32)
```

**Args:**

- **dim** (`Int`)
- **metric** (`def(Float32) -> Float32`)
- **self** (`Self`)

**Returns:**

`Self`

### `search`

```mojo
def search(self, mut sr: SearchRecord)
```

**Args:**

- **self** (`Self`)
- **sr** (`SearchRecord`)

### `box_in_search_range`

```mojo
def box_in_search_range(self, sr: SearchRecord) -> Bool
```

**Args:**

- **self** (`Self`)
- **sr** (`SearchRecord`)

**Returns:**

`Bool`

### `process_terminal_node`

```mojo
def process_terminal_node(self, mut sr: SearchRecord)
```

**Args:**

- **self** (`Self`)
- **sr** (`SearchRecord`)

### `process_terminal_node_fixedball`

```mojo
def process_terminal_node_fixedball(self, sr: SearchRecord)
```

**Args:**

- **self** (`Self`)
- **sr** (`SearchRecord`)


