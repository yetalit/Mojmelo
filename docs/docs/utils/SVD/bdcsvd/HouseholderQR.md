Mojo struct

# `HouseholderQR`

```mojo
@memory_only
struct HouseholderQR
```

## Fields

- **m_qr** (`Mat`)
- **m_hCoeffs** (`Vec`)
- **m_rows** (`Int`)
- **m_cols** (`Int`)

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

The cols x cols (== m_cols x m_cols) upper-triangular R factor, as a fresh dense copy — matches Eigen's `qrDecomp.matrixQR().topRows(diagSize).triangularView<StrictlyLower>().setZero()`.

**Args:**

- **self** (`Self`)

**Returns:**

`Mat`

### `apply_q_on_left`

```mojo
fn def apply_q_on_left(self, mut M: Mat)
```

M <- Q * M, i.e. H_0 * H_1 * ... * H_{m_cols-1} * M, applied via compact-WY panels of `block_size` reflectors at a time (2 GEMMs per panel) instead of one rank-1 update per reflector. M here is the full rows x rows (or cols x cols) matrixU/matrixV, so this is the single most expensive Householder-apply in the whole solve for rectangular inputs.

**Args:**

- **self** (`Self`)
- **M** (`Mat`)


