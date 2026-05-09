Mojo struct

# `ONE_CLASS_Q`

```mojo
@memory_only
struct ONE_CLASS_Q[k_t: Int]
```

## Aliases

- `kernel_function = kernel_linear if (k_t == LINEAR) else kernel_poly if (k_t == POLY) else kernel_rbf if (k_t == RBF) else kernel_sigmoid if (k_t == SIGMOID) else kernel_precomputed`

## Parameters

- **k_t** (`Int`)

## Fields

- **cache** (`Cache`)
- **QD** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`, `QMatrix`

## Methods

### `__init__`

```mojo
fn __init__(out self, prob: svm_problem, param: svm_parameter)
```

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `get_Q`

```mojo
fn get_Q(mut self, i: Int, _len: Int) -> Optional[UnsafePointer[Float32, MutExternalOrigin]]
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **_len** (`Int`)

**Returns:**

`Optional[UnsafePointer[Float32, MutExternalOrigin]]`

### `get_QD`

```mojo
fn get_QD(self) -> Optional[UnsafePointer[Float64, MutExternalOrigin]]
```

**Args:**

- **self** (`Self`)

**Returns:**

`Optional[UnsafePointer[Float64, MutExternalOrigin]]`

### `swap_index`

```mojo
fn swap_index(mut self, i: Int, j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)


