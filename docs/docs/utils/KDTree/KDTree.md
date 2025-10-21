Mojo struct

# `KDTree`

```mojo
@memory_only
struct KDTree[sort_results: Bool = False, rearrange: Bool = True]
```

## Aliases

- `bucketsize = 12`
- `__del__is_trivial = False`
- `__moveinit__is_trivial = False`
- `__copyinit__is_trivial = False`

## Parameters

- **sort_results** (`Bool`)
- **rearrange** (`Bool`)

## Fields

- **N** (`Int`)
- **dim** (`Int`)
- **root** (`UnsafePointer[KDTreeNode]`)
- **ind** (`List[Scalar[DType.index]]`)
- **metric** (`fn(Float32) -> Float32`)

## Implemented traits

`AnyType`, `Copyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, X: Matrix, metric: String = "euc", *, build: Bool = True)
```

**Args:**

- **X** (`Matrix`)
- **metric** (`String`)
- **build** (`Bool`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__moveinit__`

```mojo
@staticmethod
fn __moveinit__(out self, var existing: Self)
```

**Args:**

- **existing** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn __del__(var self)
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
fn build_tree_for_range(mut self, l: Int, u: Int, parent: UnsafePointer[KDTreeNode]) -> UnsafePointer[KDTreeNode]
```

**Args:**

- **self** (`Self`)
- **l** (`Int`)
- **u** (`Int`)
- **parent** (`UnsafePointer`)

**Returns:**

`UnsafePointer`

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
fn n_nearest(self, qv: NDBuffer[DType.float32, 1, origin], nn: Int, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **qv** (`NDBuffer`)
- **nn** (`Int`)
- **result** (`KDTreeResultVector`)

### `n_nearest_around_point`

```mojo
fn n_nearest_around_point(self, idxin: Int, correltime: Int, nn: Int, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **idxin** (`Int`)
- **correltime** (`Int`)
- **nn** (`Int`)
- **result** (`KDTreeResultVector`)

### `r_nearest`

```mojo
fn r_nearest(self, qv: NDBuffer[DType.float32, 1, origin], r2: Float32, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **qv** (`NDBuffer`)
- **r2** (`Float32`)
- **result** (`KDTreeResultVector`)

### `r_count`

```mojo
fn r_count(self, qv: NDBuffer[DType.float32, 1, origin], r2: Float32) -> Int
```

**Args:**

- **self** (`Self`)
- **qv** (`NDBuffer`)
- **r2** (`Float32`)

**Returns:**

`Int`

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

### `r_count_around_point`

```mojo
fn r_count_around_point(self, idxin: Int, correltime: Int, r2: Float32) -> Int
```

**Args:**

- **self** (`Self`)
- **idxin** (`Int`)
- **correltime** (`Int`)
- **r2** (`Float32`)

**Returns:**

`Int`


