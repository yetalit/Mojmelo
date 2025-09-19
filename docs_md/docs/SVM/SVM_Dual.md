Mojo struct

# `SVM_Dual`

```mojo
@memory_only
struct SVM_Dual
```

## Fields

- **lr** (`SIMD[float32, 1]`)
- **epoches** (`Int`)
- **C** (`SIMD[float32, 1]`)
- **kernel** (`String`)
- **kernel_func** (`fn(Tuple[SIMD[float32, 1], Int], Matrix, Matrix) raises -> Matrix`)
- **degree** (`Int`)
- **gamma** (`SIMD[float32, 1]`)
- **class_zero** (`Bool`)
- **k_params** (`Tuple[SIMD[float32, 1], Int]`)
- **alpha** (`Matrix`)
- **bias** (`SIMD[float32, 1]`)
- **X** (`Matrix`)
- **y** (`Matrix`)

## Implemented traits

`AnyType`, `CVM`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, learning_rate: SIMD[float32, 1] = 0.001, n_iters: Int = 1000, C: SIMD[float32, 1] = 1, kernel: String = "poly", degree: Int = 2, gamma: SIMD[float32, 1] = -1, class_zero: Bool = False)
```

**Args:**

- **learning_rate** (`SIMD`)
- **n_iters** (`Int`)
- **C** (`SIMD`)
- **kernel** (`String`)
- **degree** (`Int`)
- **gamma** (`SIMD`)
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


