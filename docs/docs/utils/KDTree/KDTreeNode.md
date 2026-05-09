Mojo struct

# `KDTreeNode`

```mojo
@memory_only
struct KDTreeNode[metric: String = "euc"]
```

## Aliases

- `metric_func = Squared if (metric == String("euc")) else Abs`

## Parameters

- **metric** (`String`)

## Fields

- **cut_dim** (`Int`)
- **cut_val** (`Float32`)
- **cut_val_left** (`Float32`)
- **cut_val_right** (`Float32`)
- **l** (`Int`)
- **u** (`Int`)
- **box** (`List[interval]`)
- **left** (`Optional[UnsafePointer[KDTreeNode[metric], MutAnyOrigin]]`)
- **right** (`Optional[UnsafePointer[KDTreeNode[metric], MutAnyOrigin]]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__(out self, dim: Int)
```

**Args:**

- **dim** (`Int`)
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

**Raises:**

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


