Mojo struct

# `KDTreeBoruvka`

```mojo
@memory_only
struct KDTreeBoruvka
```

## Fields

- **data** (`UnsafePointer[Float32, MutAnyOrigin]`)
- **kdtree** (`KDTree[True]`)
- **n** (`Int`)
- **dim** (`Int`)
- **leaf_size** (`Int`)
- **nodes** (`List[NodeData]`)
- **core_dist** (`UnsafePointer[Float32, MutAnyOrigin]`)
- **build_idx** (`List[Scalar[DType.int]]`)
- **proj_buf** (`List[Float32]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
def __init__(out self, data: Matrix, min_samples: Int, leaf_size: Int, search_depth: Int)
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

### `__del__`

```mojo
def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `left`

```mojo
def left(self, i: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Int`

### `right`

```mojo
def right(self, i: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Int`

### `ensure_node`

```mojo
def ensure_node(mut self, i: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

### `choose_split_dim`

```mojo
def choose_split_dim(self, start: Int, end: Int, idx: List[Scalar[DType.int]]) -> Scalar[DType.int]
```

**Args:**

- **self** (`Self`)
- **start** (`Int`)
- **end** (`Int`)
- **idx** (`List`)

**Returns:**

`Scalar`

### `build_node`

```mojo
def build_node(mut self, node: Int, start: Int, end: Int)
```

**Args:**

- **self** (`Self`)
- **node** (`Int`)
- **start** (`Int`)
- **end** (`Int`)


