Mojo struct

# `SVM_Primal`

```mojo
@memory_only
struct SVM_Primal
```

## Fields

- **lr** (`SIMD[float32, 1]`)
- **lambda_param** (`SIMD[float32, 1]`)
- **n_iters** (`Int`)
- **class_zero** (`Bool`)
- **weights** (`Matrix`)
- **bias** (`SIMD[float32, 1]`)

## Implemented traits

`AnyType`, `CVM`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, learning_rate: SIMD[float32, 1] = 0.001, lambda_param: SIMD[float32, 1] = 0.01, n_iters: Int = 1000, class_zero: Bool = False)
```

**Args:**

- **learning_rate** (`SIMD`)
- **lambda_param** (`SIMD`)
- **n_iters** (`Int`)
- **class_zero** (`Bool`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: Matrix)
```

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn predict(self, X: Matrix) -> Matrix
```

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`

**Raises:**


