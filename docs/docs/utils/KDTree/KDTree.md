Mojo struct

# `KDTree`

```mojo
@memory_only
struct KDTree[sort_results: Bool = False, rearrange: Bool = True]
```

## Aliases

- `bucketsize = 12`

## Parameters

- **sort_results** (`Bool`)
- **rearrange** (`Bool`)

## Fields

- **N** (`Int`)
- **dim** (`Int`)
- **root** (`UnsafePointer[KDTreeNode, MutAnyOrigin]`)
- **ind** (`List[Scalar[DType.int]]`)
- **metric** (`def(Float32) -> Float32`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
def __init__(out self, X: Matrix, metric: String = "euc", *, build: Bool = True)
```

**Args:**

- **X** (`Matrix`)
- **metric** (`String`)
- **build** (`Bool`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __init__(out self, *, deinit take: Self)
```

**Args:**

- **take** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `build_tree`

```mojo
def build_tree(mut self)
```

**Args:**

- **self** (`Self`)

**Raises:**

### `build_tree_for_range`

```mojo
def build_tree_for_range(mut self, l: Int, u: Int, parent: UnsafePointer[KDTreeNode, MutAnyOrigin]) -> UnsafePointer[KDTreeNode, MutAnyOrigin]
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
def spread_in_coordinate(self, c: Int, l: Int, u: Int, mut interv: interval)
```

**Args:**

- **self** (`Self`)
- **c** (`Int`)
- **l** (`Int`)
- **u** (`Int`)
- **interv** (`interval`)

### `select_on_coordinate`

```mojo
def select_on_coordinate(mut self, c: Int, k: Int, var l: Int, var u: Int)
```

**Args:**

- **self** (`Self`)
- **c** (`Int`)
- **k** (`Int`)
- **l** (`Int`)
- **u** (`Int`)

### `select_on_coordinate_value`

```mojo
def select_on_coordinate_value(mut self, c: Int, alpha: Float32, l: Int, u: Int) -> Int
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
def n_nearest(mut self, qv: Span[Float32, MutAnyOrigin], nn: Int, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **qv** (`Span`)
- **nn** (`Int`)
- **result** (`KDTreeResultVector`)

### `n_nearest_around_point`

```mojo
def n_nearest_around_point(mut self, idxin: Int, correltime: Int, nn: Int, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **idxin** (`Int`)
- **correltime** (`Int`)
- **nn** (`Int`)
- **result** (`KDTreeResultVector`)

### `r_nearest`

```mojo
def r_nearest(mut self, qv: Span[Float32, MutAnyOrigin], r2: Float32, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **qv** (`Span`)
- **r2** (`Float32`)
- **result** (`KDTreeResultVector`)

### `r_count`

```mojo
def r_count(mut self, qv: Span[Float32, MutAnyOrigin], r2: Float32) -> Int
```

**Args:**

- **self** (`Self`)
- **qv** (`Span`)
- **r2** (`Float32`)

**Returns:**

`Int`

### `r_nearest_around_point`

```mojo
def r_nearest_around_point(mut self, idxin: Int, correltime: Int, r2: Float32, mut result: KDTreeResultVector)
```

**Args:**

- **self** (`Self`)
- **idxin** (`Int`)
- **correltime** (`Int`)
- **r2** (`Float32`)
- **result** (`KDTreeResultVector`)

### `r_count_around_point`

```mojo
def r_count_around_point(mut self, idxin: Int, correltime: Int, r2: Float32) -> Int
```

**Args:**

- **self** (`Self`)
- **idxin** (`Int`)
- **correltime** (`Int`)
- **r2** (`Float32`)

**Returns:**

`Int`


