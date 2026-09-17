Mojo struct

# `ColPivHouseholderQR`

```mojo
@memory_only
struct ColPivHouseholderQR
```

## Fields

- **m_qr** (`Mat`)
- **m_hCoeffs** (`Vec`)
- **m_colsPermutation** (`IVec`)
- **m_rows** (`Int`)
- **m_cols** (`Int`)
- **m_maxPivot** (`RealScalar`)

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

### `matrixR`

```mojo
fn def matrixR(self) -> Mat
```

The cols x cols (== m_cols x m_cols) upper-triangular R factor of A*P, as a fresh dense copy.

**Args:**

- **self** (`Self`)

**Returns:**

`Mat`

### `apply_q_on_left`

```mojo
fn def apply_q_on_left(self, mut M: Mat)
```

M <- Q * M, i.e. H_0 * H_1 * ... * H_{m_cols-1} * M — reflectors applied in reverse order, same pattern as UpperBidiagonalization.apply_u_on_left / bdcsvd.HouseholderQR.

**Args:**

- **self** (`Self`)
- **M** (`Mat`)


