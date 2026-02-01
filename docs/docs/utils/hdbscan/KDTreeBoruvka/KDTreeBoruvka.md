Mojo struct

# `KDTreeBoruvka`

```mojo
@memory_only
struct KDTreeBoruvka
```

## Aliases

- `__del__is_trivial = False`

## Fields

- **data** (`UnsafePointer[Float32, MutAnyOrigin]`)
- **kdtree** (`KDTree[True]`)
- **n** (`Int`)
- **dim** (`Int`)
- **leaf_size** (`Int`)
- **nodes** (`List[NodeData]`)
- **core_dist** (`UnsafePointer[Float32, MutAnyOrigin]`)
- **build_idx** (`List[Scalar[DType.index]]`)
- **proj_buf** (`List[Float32]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, data: Matrix, min_samples: Int, leaf_size: Int, search_deepness_coef: Int)
```

**Args:**

- **data** (`Matrix`)
- **min_samples** (`Int`)
- **leaf_size** (`Int`)
- **search_deepness_coef** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__del__`

```mojo
fn __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `left`

```mojo
fn left(self, i: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Int`

### `right`

```mojo
fn right(self, i: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Int`

### `ensure_node`

```mojo
fn ensure_node(mut self, i: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

### `choose_split_dim`

```mojo
fn choose_split_dim(self, start: Int, end: Int, idx: List[Scalar[DType.index]]) -> Int
```

**Args:**

- **self** (`Self`)
- **start** (`Int`)
- **end** (`Int`)
- **idx** (`List`)

**Returns:**

`Int`

### `build_node`

```mojo
fn build_node(mut self, node: Int, start: Int, end: Int)
```

**Args:**

- **self** (`Self`)
- **node** (`Int`)
- **start** (`Int`)
- **end** (`Int`)


