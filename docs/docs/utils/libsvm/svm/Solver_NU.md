Mojo struct

# `Solver_NU`

```mojo
@memory_only
struct Solver_NU
```

## Aliases

- `LOWER_BOUND = Int8(0)`
- `UPPER_BOUND = Int8(1)`
- `FREE = Int8(2)`

## Fields

- **si** (`SolutionInfo`)
- **active_size** (`Int`)
- **y** (`UnsafePointer[Int8, MutUntrackedOrigin]`)
- **G** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **alpha_status** (`UnsafePointer[Int8, MutUntrackedOrigin]`)
- **alpha** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **QD** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **eps** (`Float64`)
- **Cp** (`Float64`)
- **Cn** (`Float64`)
- **p** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **active_set** (`UnsafePointer[Int, MutUntrackedOrigin]`)
- **G_bar** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **l** (`Int`)
- **unshrink** (`Bool`)

## Implemented traits

`AnyType`, `ImplicitlyDeletable`

## Methods

### `__init__`

```mojo
fn def __init__(out self)
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `get_C`

```mojo
fn def get_C(self, i: Int) -> Float64
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Float64`

### `update_alpha_status`

```mojo
fn def update_alpha_status(self, i: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

### `is_upper_bound`

```mojo
fn def is_upper_bound(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `is_lower_bound`

```mojo
fn def is_lower_bound(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `is_free`

```mojo
fn def is_free(self, i: Int) -> Bool
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Bool`

### `swap_index`

```mojo
fn def swap_index[QM: QMatrix](self, mut Q: QM, i: Int, j: Int)
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
fn def reconstruct_gradient[QM: QMatrix](self, mut Q: QM)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **Q** (`QM`)

### `Solve`

```mojo
fn def Solve[QM: QMatrix](mut self, l: Int, mut Q: QM, p_: Optional[UnsafePointer[Float64, MutUntrackedOrigin]], y_: Optional[UnsafePointer[Int8, MutUntrackedOrigin]], alpha_: UnsafePointer[Float64, MutUntrackedOrigin], Cp: Float64, Cn: Float64, eps: Float64, si: SolutionInfo, shrinking: Int)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **l** (`Int`)
- **Q** (`QM`)
- **p_** (`Optional[UnsafePointer[Float64, MutUntrackedOrigin]]`)
- **y_** (`Optional[UnsafePointer[Int8, MutUntrackedOrigin]]`)
- **alpha_** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **Cp** (`Float64`)
- **Cn** (`Float64`)
- **eps** (`Float64`)
- **si** (`SolutionInfo`)
- **shrinking** (`Int`)

### `select_working_set`

```mojo
fn def select_working_set[QM: QMatrix](self, mut Q: QM, mut out_i: Int, mut out_j: Int) -> Int
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
fn def be_shrunk(self, i: Int, Gmax1: Float64, Gmax2: Float64, Gmax3: Float64, Gmax4: Float64) -> Bool
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
fn def do_shrinking[QM: QMatrix](mut self, mut Q: QM)
```

**Parameters:**

- **QM** (`QMatrix`)

**Args:**

- **self** (`Self`)
- **Q** (`QM`)

### `calculate_rho`

```mojo
fn def calculate_rho(mut self) -> Float64
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float64`


