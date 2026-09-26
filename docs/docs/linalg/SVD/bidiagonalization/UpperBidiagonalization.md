Mojo struct

# `UpperBidiagonalization`

```mojo
@memory_only
struct UpperBidiagonalization
```

## Fields

- **m_householder** (`Mat`)
- **m_diag** (`Vec`)
- **m_superdiag** (`Vec`)
- **m_rows** (`Int`)
- **m_cols** (`Int`)
- **m_isInitialized** (`Bool`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self)
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `compute`

```mojo
fn def compute(mut self, A: Mat)
```

**Args:**

- **self** (`Self`)
- **A** (`Mat`)

### `compute_unblocked`

```mojo
fn def compute_unblocked(mut self, A: Mat)
```

**Args:**

- **self** (`Self`)
- **A** (`Mat`)

### `bidiagonal_diagonal`

```mojo
fn def bidiagonal_diagonal(self) -> Vec
```

**Args:**

- **self** (`Self`)

**Returns:**

`Vec`

### `bidiagonal_superdiagonal`

```mojo
fn def bidiagonal_superdiagonal(self) -> Vec
```

**Args:**

- **self** (`Self`)

**Returns:**

`Vec`

### `apply_u_on_left`

```mojo
fn def apply_u_on_left(self, mut M: Mat)
```

**Args:**

- **self** (`Self`)
- **M** (`Mat`)

### `apply_v_on_left`

```mojo
fn def apply_v_on_left(self, mut M: Mat)
```

**Args:**

- **self** (`Self`)
- **M** (`Mat`)


