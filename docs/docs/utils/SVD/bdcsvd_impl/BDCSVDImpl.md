Mojo struct

# `BDCSVDImpl`

```mojo
@memory_only
struct BDCSVDImpl
```

## Fields

- **m_naiveU** (`Mat`)
- **m_naiveV** (`Mat`)
- **m_computed** (`Mat`)
- **m_workspace** (`Vec`)
- **m_workspaceI** (`IVec`)
- **m_baseSvdU** (`JacobiSVD`)
- **m_baseSvdUV** (`JacobiSVD`)
- **m_algoswap** (`Int`)
- **m_compU** (`Bool`)
- **m_compV** (`Bool`)
- **m_numIters** (`Int`)
- **m_info** (`ComputationInfo`)

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

### `algoSwap`

```mojo
fn def algoSwap(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `setAlgoSwap`

```mojo
fn def setAlgoSwap(mut self, s: Int)
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

### `numIters`

```mojo
fn def numIters(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `naiveU`

```mojo
fn def naiveU(mut self) -> Mat
```

**Args:**

- **self** (`Self`)

**Returns:**

`Mat`

### `naiveV`

```mojo
fn def naiveV(mut self) -> Mat
```

**Args:**

- **self** (`Self`)

**Returns:**

`Mat`

### `computed`

```mojo
fn def computed(mut self) -> Mat
```

**Args:**

- **self** (`Self`)

**Returns:**

`Mat`

### `allocate`

```mojo
fn def allocate(mut self, diagSize: Int, compU: Bool, compV: Bool)
```

**Args:**

- **self** (`Self`)
- **diagSize** (`Int`)
- **compU** (`Bool`)
- **compV** (`Bool`)

### `splitNegligibleSuperdiagonal`

```mojo
fn def splitNegligibleSuperdiagonal(mut self, n: Int)
```

**Args:**

- **self** (`Self`)
- **n** (`Int`)

### `structured_update`

```mojo
fn def structured_update(mut self, mut A: Mat, B: Mat, n1: Int)
```

**Args:**

- **self** (`Self`)
- **A** (`Mat`)
- **B** (`Mat`)
- **n1** (`Int`)

### `computeBaseCase`

```mojo
fn def computeBaseCase[V: Bool](mut self, n: Int, firstCol: Int, firstRowW: Int, firstColW: Int, shift: Int)
```

**Parameters:**

- **V** (`Bool`)

**Args:**

- **self** (`Self`)
- **n** (`Int`)
- **firstCol** (`Int`)
- **firstRowW** (`Int`)
- **firstColW** (`Int`)
- **shift** (`Int`)

### `divide`

```mojo
fn def divide(mut self, firstCol: Int, lastCol: Int, firstRowW: Int, firstColW: Int, shift: Int)
```

**Args:**

- **self** (`Self`)
- **firstCol** (`Int`)
- **lastCol** (`Int`)
- **firstRowW** (`Int`)
- **firstColW** (`Int`)
- **shift** (`Int`)

### `computeSVDofM`

```mojo
fn def computeSVDofM(mut self, firstCol: Int, n: Int, mut U: Mat, mut singVals: Vec, mut V: Mat)
```

**Args:**

- **self** (`Self`)
- **firstCol** (`Int`)
- **n** (`Int`)
- **U** (`Mat`)
- **singVals** (`Vec`)
- **V** (`Mat`)

### `computeSingVals`

```mojo
fn def computeSingVals(mut self, col0: Vec, diag: Vec, perm: IVec, mut singVals: Vec, mut shifts: Vec, mut mus: Vec)
```

**Args:**

- **self** (`Self`)
- **col0** (`Vec`)
- **diag** (`Vec`)
- **perm** (`IVec`)
- **singVals** (`Vec`)
- **shifts** (`Vec`)
- **mus** (`Vec`)

### `perturbCol0`

```mojo
fn def perturbCol0(mut self, col0: Vec, diag: Vec, perm: IVec, singVals: Vec, shifts: Vec, mus: Vec, mut zhat: Vec)
```

**Args:**

- **self** (`Self`)
- **col0** (`Vec`)
- **diag** (`Vec`)
- **perm** (`IVec`)
- **singVals** (`Vec`)
- **shifts** (`Vec`)
- **mus** (`Vec`)
- **zhat** (`Vec`)

### `computeSingVecs`

```mojo
fn def computeSingVecs(mut self, zhat: Vec, diag: Vec, perm: IVec, singVals: Vec, shifts: Vec, mus: Vec, mut U: Mat, mut V: Mat)
```

**Args:**

- **self** (`Self`)
- **zhat** (`Vec`)
- **diag** (`Vec`)
- **perm** (`IVec`)
- **singVals** (`Vec`)
- **shifts** (`Vec`)
- **mus** (`Vec`)
- **U** (`Mat`)
- **V** (`Mat`)

### `deflation43`

```mojo
fn def deflation43(mut self, firstCol: Int, shift: Int, i: Int, size: Int)
```

**Args:**

- **self** (`Self`)
- **firstCol** (`Int`)
- **shift** (`Int`)
- **i** (`Int`)
- **size** (`Int`)

### `deflation44`

```mojo
fn def deflation44(mut self, firstColu: Int, firstColm: Int, firstRowW: Int, firstColW: Int, i: Int, j: Int, size: Int)
```

**Args:**

- **self** (`Self`)
- **firstColu** (`Int`)
- **firstColm** (`Int`)
- **firstRowW** (`Int`)
- **firstColW** (`Int`)
- **i** (`Int`)
- **j** (`Int`)
- **size** (`Int`)

### `deflation`

```mojo
fn def deflation(mut self, firstCol: Int, lastCol: Int, k: Int, firstRowW: Int, firstColW: Int, shift: Int)
```

**Args:**

- **self** (`Self`)
- **firstCol** (`Int`)
- **lastCol** (`Int`)
- **k** (`Int`)
- **firstRowW** (`Int`)
- **firstColW** (`Int`)
- **shift** (`Int`)


