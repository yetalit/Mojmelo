Mojo struct

# `Solver`

```mojo
@memory_only
struct Solver
```

## Aliases

- `LOWER_BOUND = 0`
- `UPPER_BOUND = 1`
- `FREE = 2`
- `__del__is_trivial = True`

## Fields

- **active_size** (`Int`)
- **y** (`UnsafePointer[Int8]`)
- **G** (`UnsafePointer[Float64]`)
- **alpha_status** (`UnsafePointer[Int8]`)
- **alpha** (`UnsafePointer[Float64]`)
- **QD** (`UnsafePointer[Float64]`)
- **eps** (`Float64`)
- **Cp** (`Float64`)
- **Cn** (`Float64`)
- **p** (`UnsafePointer[Float64]`)
- **active_set** (`UnsafePointer[Scalar[DType.index]]`)
- **G_bar** (`UnsafePointer[Float64]`)
- **l** (`Int`)
- **unshrink** (`Bool`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self)
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `get_C`

```mojo
fn get_C(self, i: Int) -> Float64
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Float64`

### `update_alpha_status`

```mojo
fn update_alpha_status(self, i: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

### `is_upper_bound`

```mojo
fn is_upper_bound(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `is_lower_bound`

```mojo
fn is_lower_bound(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `is_free`

```mojo
fn is_free(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `swap_index`

```mojo
fn swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **Q** (`QM`)
- **i** (`Int`)
- **j** (`Int`)

### `reconstruct_gradient`

```mojo
fn reconstruct_gradient[QM: QMatrix](self, mut Q: QM)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **Q** (`QM`)

### `Solve`

```mojo
fn Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: UnsafePointer[Float64], y_: UnsafePointer[Int8], alpha_: UnsafePointer[Float64], Cp: Float64, Cn: Float64, eps: Float64, si: UnsafePointer[SolutionInfo], shrinking: Int)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **l** (`Int`)
- **Q** (`QM`)
- **p_** (`UnsafePointer`)
- **y_** (`UnsafePointer`)
- **alpha_** (`UnsafePointer`)
- **Cp** (`Float64`)
- **Cn** (`Float64`)
- **eps** (`Float64`)
- **si** (`UnsafePointer`)
- **shrinking** (`Int`)

### `select_working_set`

```mojo
fn select_working_set[QM: QMatrix](self, mut Q: QM, mut out_i: Int, mut out_j: Int) -> Int
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **Q** (`QM`)
- **out_i** (`Int`)
- **out_j** (`Int`)

**Returns:**

`Int`

### `be_shrunk`

```mojo
fn be_shrunk(self, i: Int, Gmax1: Float64, Gmax2: Float64) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **Gmax1** (`Float64`)
- **Gmax2** (`Float64`)

**Returns:**

`Bool`

### `do_shrinking`

```mojo
fn do_shrinking[QM: QMatrix](mut self, mut Q: QM)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **Q** (`QM`)

### `calculate_rho`

```mojo
fn calculate_rho(self) -> Float64
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float64`


