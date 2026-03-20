Mojo struct

# `Solver_NU`

```mojo
@memory_only
struct Solver_NU
```

## Aliases

- `LOWER_BOUND = 0`
- `UPPER_BOUND = 1`
- `FREE = 2`

## Fields

- **si** (`SolutionInfo`)
- **active_size** (`Int`)
- **y** (`UnsafePointer[Int8, MutExternalOrigin]`)
- **G** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **alpha_status** (`UnsafePointer[Int8, MutExternalOrigin]`)
- **alpha** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **QD** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **eps** (`Float64`)
- **Cp** (`Float64`)
- **Cn** (`Float64`)
- **p** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **active_set** (`UnsafePointer[Scalar[DType.int], MutExternalOrigin]`)
- **G_bar** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **l** (`Int`)
- **unshrink** (`Bool`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
def __init__(out self)
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `get_C`

```mojo
def get_C(self, i: Int) -> Float64
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Float64`

### `update_alpha_status`

```mojo
def update_alpha_status(self, i: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

### `is_upper_bound`

```mojo
def is_upper_bound(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `is_lower_bound`

```mojo
def is_lower_bound(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `is_free`

```mojo
def is_free(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `swap_index`

```mojo
def swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int)
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
def reconstruct_gradient[QM: QMatrix](self, mut Q: QM)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **Q** (`QM`)

### `Solve`

```mojo
def Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: UnsafePointer[Float64, MutExternalOrigin], y_: UnsafePointer[Int8, MutExternalOrigin], alpha_: UnsafePointer[Float64, MutExternalOrigin], Cp: Float64, Cn: Float64, eps: Float64, si: SolutionInfo, shrinking: Int)
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
- **si** (`SolutionInfo`)
- **shrinking** (`Int`)

### `select_working_set`

```mojo
def select_working_set[QM: QMatrix](self, mut Q: QM, mut out_i: Int, mut out_j: Int) -> Int
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
def be_shrunk(self, i: Int, Gmax1: Float64, Gmax2: Float64, Gmax3: Float64, Gmax4: Float64) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **Gmax1** (`Float64`)
- **Gmax2** (`Float64`)
- **Gmax3** (`Float64`)
- **Gmax4** (`Float64`)

**Returns:**

`Bool`

### `do_shrinking`

```mojo
def do_shrinking[QM: QMatrix](mut self, mut Q: QM)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **Q** (`QM`)

### `calculate_rho`

```mojo
def calculate_rho(mut self) -> Float64
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float64`


