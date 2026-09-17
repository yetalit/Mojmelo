Mojo struct

# `BDCSVD`

```mojo
@memory_only
struct BDCSVD
```

## Fields

- **m_impl** (`BDCSVDImpl`)
- **m_isTranspose** (`Bool`)
- **m_computeU** (`Bool`)
- **m_computeV** (`Bool`)
- **m_computeThinU** (`Bool`)
- **m_computeThinV** (`Bool`)
- **m_matrixU** (`Mat`)
- **m_matrixV** (`Mat`)
- **m_singularValues** (`Vec`)
- **m_nonzeroSingularValues** (`Int`)
- **m_info** (`ComputationInfo`)
- **m_diagSize** (`Int`)
- **m_numIters** (`Int`)
- **smallSvd** (`JacobiSVD`)
- **m_useQrDecomp** (`Bool`)
- **qrDecomp** (`HouseholderQR`)

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

### `setSwitchSize`

```mojo
fn def setSwitchSize(mut self, s: Int)
```

**Args:**

- **self** (`Self`)
- **s** (`Int`)

### `info`

```mojo
fn def info(self) -> ComputationInfo
```

**Args:**

- **self** (`Self`)

**Returns:**

`ComputationInfo`

### `singularValues`

```mojo
fn def singularValues(self) -> Vec
```

**Args:**

- **self** (`Self`)

**Returns:**

`Vec`

### `matrixU`

```mojo
fn def matrixU(self) -> Mat
```

**Args:**

- **self** (`Self`)

**Returns:**

`Mat`

### `matrixV`

```mojo
fn def matrixV(self) -> Mat
```

**Args:**

- **self** (`Self`)

**Returns:**

`Mat`

### `nonzeroSingularValues`

```mojo
fn def nonzeroSingularValues(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `allocate`

```mojo
fn def allocate(mut self, rows: Int, cols: Int, computeU: Bool, computeV: Bool, thinU: Bool = False, thinV: Bool = False)
```

**Args:**

- **self** (`Self`)
- **rows** (`Int`)
- **cols** (`Int`)
- **computeU** (`Bool`)
- **computeV** (`Bool`)
- **thinU** (`Bool`)
- **thinV** (`Bool`)

### `extractSingularValues`

```mojo
fn def extractSingularValues(mut self, scale: Float64)
```

**Args:**

- **self** (`Self`)
- **scale** (`Float64`)

### `compute_bidiagonal`

```mojo
fn def compute_bidiagonal(mut self, diag: Vec, superdiag: Vec, computeU: Bool, computeV: Bool) -> ComputationInfo
```

**Args:**

- **self** (`Self`)
- **diag** (`Vec`)
- **superdiag** (`Vec`)
- **computeU** (`Bool`)
- **computeV** (`Bool`)

**Returns:**

`ComputationInfo`

### `compute`

```mojo
fn def compute(mut self, A: Mat, computeU: Bool, computeV: Bool, thinU: Bool = False, thinV: Bool = False) -> ComputationInfo
```

**Args:**

- **self** (`Self`)
- **A** (`Mat`)
- **computeU** (`Bool`)
- **computeV** (`Bool`)
- **thinU** (`Bool`)
- **thinV** (`Bool`)

**Returns:**

`ComputationInfo`


