Mojo struct

# `KDTree`

```mojo
@memory_only
struct KDTree[sort_results: Bool = False, rearrange: Bool = True, metric: String = "euc"]
```

## Aliases

- `bucketsize = 12`

## Parameters

- **sort_results** (`Bool`)
- **rearrange** (`Bool`)
- **metric** (`String`)

## Fields

- **N** (`Int`)
- **dim** (`Int`)
- **root** (`Optional[UnsafePointer[KDTreeNode[metric], MutAnyOrigin]]`)
- **ind** (`List[Int]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__(out self, X: Matrix, *, build: Bool = True)
```

**Args:**

- **X** (`Matrix`)
- **build** (`Bool`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __init__(out self, *, deinit take: Self)
```

**Args:**

- **take** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `build_tree`

```mojo
fn build_tree(mut self)
```

**Args:**

- **self** (`Self`)

**Raises:**

### `build_tree_for_range`

```mojo
fn build_tree_for_range(mut self, l: Int, u: Int, parent: Optional[UnsafePointer[KDTreeNode[metric], MutAnyOrigin]]) -> Optional[UnsafePointer[KDTreeNode[metric], MutAnyOrigin]]
```

**Args:**

- **self** (`Self`)
- **l** (`Int`)
- **u** (`Int`)
- **parent** (`Optional[UnsafePointer[KDTreeNode[metric], MutAnyOrigin]]`)

**Returns:**

`Optional[UnsafePointer[KDTreeNode[metric], MutAnyOrigin]]`

**Raises:**

### `spread_in_coordinate`

```mojo
fn spread_in_coordinate(self, c: Int, l: Int, u: Int, mut interv: interval)
```

**Args:**

- **self** (`Self`)
- **c** (`Int`)
- **l** (`Int`)
- **u** (`Int`)
- **interv** (`interval`)

### `select_on_coordinate`

```mojo
fn select_on_coordinate(mut self, c: Int, k: Int, var l: Int, var u: Int)
```

**Args:**

- **self** (`Self`)
- **c** (`Int`)
- **k** (`Int`)
- **l** (`Int`)
- **u** (`Int`)

### `select_on_coordinate_value`

```mojo
fn select_on_coordinate_value(mut self, c: Int, alpha: Float32, l: Int, u: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **c** (`Int`)
- **alpha** (`Float32`)
- **l** (`Int`)
- **u** (`Int`)

**Returns:**

`Int`

### `n_nearest`

```mojo
fn n_nearest(mut self, qv: Span[Float32, MutAnyOrigin], nn: Int, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **qv** (`Span[Float32, MutAnyOrigin]`)
- **nn** (`Int`)
- **result** (`KDTreeResultVector`)

**Raises:**

### `n_nearest_around_point`

```mojo
fn n_nearest_around_point(mut self, idxin: Int, correltime: Int, nn: Int, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **idxin** (`Int`)
- **correltime** (`Int`)
- **nn** (`Int`)
- **result** (`KDTreeResultVector`)

**Raises:**

### `r_nearest`

```mojo
fn r_nearest(mut self, qv: Span[Float32, MutAnyOrigin], r2: Float32, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **qv** (`Span[Float32, MutAnyOrigin]`)
- **r2** (`Float32`)
- **result** (`KDTreeResultVector`)

**Raises:**

### `r_count`

```mojo
fn r_count(mut self, qv: Span[Float32, MutAnyOrigin], r2: Float32) -> Int
```

**Args:**

- **self** (`Self`)
- **qv** (`Span[Float32, MutAnyOrigin]`)
- **r2** (`Float32`)

**Returns:**

`Int`

**Raises:**

### `r_nearest_around_point`

```mojo
fn r_nearest_around_point(mut self, idxin: Int, correltime: Int, r2: Float32, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **idxin** (`Int`)
- **correltime** (`Int`)
- **r2** (`Float32`)
- **result** (`KDTreeResultVector`)

**Raises:**

### `r_count_around_point`

```mojo
fn r_count_around_point(mut self, idxin: Int, correltime: Int, r2: Float32) -> Int
```

**Args:**

- **self** (`Self`)
- **idxin** (`Int`)
- **correltime** (`Int`)
- **r2** (`Float32`)

**Returns:**

`Int`

**Raises:**


