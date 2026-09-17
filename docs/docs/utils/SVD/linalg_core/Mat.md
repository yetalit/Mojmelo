Mojo struct

# `Mat`

```mojo
@memory_only
struct Mat
```

## Fields

- **data** (`Pointer[Float64, MutUntrackedOrigin]`)
- **nrows** (`Int`)
- **ncols** (`Int`)
- **row_stride** (`Int`)
- **col_stride** (`Int`)
- **size** (`Int`)
- **owns** (`Bool`)

## Implemented traits

`AnyType`, `Copyable`, `Deinitable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, rows: Int, cols: Int)
```

**Args:**

- **rows** (`Int`)
- **cols** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, data: Pointer[Float64, MutUntrackedOrigin], rows: Int, cols: Int, row_stride: Int, col_stride: Int)
```

**Args:**

- **data** (`Pointer[Float64, MutUntrackedOrigin]`)
- **rows** (`Int`)
- **cols** (`Int`)
- **row_stride** (`Int`)
- **col_stride** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, *, copy: Self)
```

**Args:**

- **copy** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

### `__deinit__`

```mojo
fn def __deinit__(deinit self)
```

**Args:**

- **self** (`Self`)

### `__getitem__`

```mojo
fn def __getitem__(self, i: Int, j: Int) -> RealScalar
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

**Returns:**

`RealScalar`

### `__setitem__`

```mojo
fn def __setitem__(mut self, i: Int, j: Int, v: Float64)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)
- **v** (`Float64`)

### `rows`

```mojo
fn def rows(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `cols`

```mojo
fn def cols(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `block`

```mojo
fn def block(self, i: Int, j: Int, rows: Int, cols: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)
- **rows** (`Int`)
- **cols** (`Int`)

**Returns:**

`Self`

### `col`

```mojo
fn def col(self, j: Int) -> Vec
```

**Args:**

- **self** (`Self`)
- **j** (`Int`)

**Returns:**

`Vec`

### `row`

```mojo
fn def row(self, i: Int) -> Vec
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Vec`

### `diagonal`

```mojo
fn def diagonal(self, k: Int = Int(0)) -> Vec
```

**Args:**

- **self** (`Self`)
- **k** (`Int`)

**Returns:**

`Vec`

### `copyFrom`

```mojo
fn def copyFrom(self, other: Self)
```

**Args:**

- **self** (`Self`)
- **other** (`Self`)

### `setZero`

```mojo
fn def setZero(self)
```

**Args:**

- **self** (`Self`)

### `cwiseAbsMax`

```mojo
fn def cwiseAbsMax(self) -> RealScalar
```

**Args:**

- **self** (`Self`)

**Returns:**

`RealScalar`

### `swap_cols`

```mojo
fn def swap_cols(mut self, a: Int, b: Int)
```

**Args:**

- **self** (`Self`)
- **a** (`Int`)
- **b** (`Int`)

### `zeros`

```mojo
@staticmethod
fn def zeros(rows: Int, cols: Int) -> Self
```

**Args:**

- **rows** (`Int`)
- **cols** (`Int`)

**Returns:**

`Self`


