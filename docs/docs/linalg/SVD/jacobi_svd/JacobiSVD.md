Mojo struct

# `JacobiSVD`

```mojo
@memory_only
struct JacobiSVD
```

## Fields

- **compute_v** (`Bool`)
- **u** (`Mat`)
- **v** (`Mat`)
- **sing_vals** (`Vec`)
- **status** (`ComputationInfo`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, compute_v: Bool)
```

**Args:**

- **compute_v** (`Bool`)
- **self** (`Self`)

**Returns:**

`Self`

### `compute`

```mojo
fn def compute(mut self, m: Mat, thinU: Bool = False, thinV: Bool = False)
```

**Args:**

- **self** (`Self`)
- **m** (`Mat`)
- **thinU** (`Bool`)
- **thinV** (`Bool`)

### `info`

```mojo
fn def info(self) -> ComputationInfo
```

**Args:**

- **self** (`Self`)

**Returns:**

`ComputationInfo`

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

### `singularValues`

```mojo
fn def singularValues(self) -> Vec
```

**Args:**

- **self** (`Self`)

**Returns:**

`Vec`


