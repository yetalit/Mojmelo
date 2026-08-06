Mojo struct

# `KDTreeBoruvka`

```mojo
@memory_only
struct KDTreeBoruvka
```

## Fields

- **data** (`Pointer[Float32, MutUntrackedOrigin]`)
- **kdtree** (`KDTree[True]`)
- **n** (`Int`)
- **dim** (`Int`)
- **leaf_size** (`Int`)
- **nodes** (`List[NodeData]`)
- **core_dist** (`Pointer[Float32, MutUntrackedOrigin]`)
- **build_idx** (`List[Int]`)
- **proj_buf** (`List[Float32]`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, data: Matrix, min_samples: Int, leaf_size: Int, search_depth: Int)
```

**Args:**

- **data** (`Matrix`)
- **min_samples** (`Int`)
- **leaf_size** (`Int`)
- **search_depth** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__deinit__`

```mojo
fn def __deinit__(deinit self)
```

**Args:**

- **self** (`Self`)

### `left`

```mojo
fn def left(self, i: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Int`

### `right`

```mojo
fn def right(self, i: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Int`

### `ensure_node`

```mojo
fn def ensure_node(mut self, i: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

### `choose_split_dim`

```mojo
fn def choose_split_dim(self, start: Int, end: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **start** (`Int`)
- **end** (`Int`)

**Returns:**

`Int`

### `build_node`

```mojo
fn def build_node(mut self, node: Int, start: Int, end: Int)
```

**Args:**

- **self** (`Self`)
- **node** (`Int`)
- **start** (`Int`)
- **end** (`Int`)


