Mojo struct

# `SVR_Q`

```mojo
@memory_only
struct SVR_Q[k_t: Int]
```

## Aliases

- `kernel_function = kernel_linear if (k_t == LINEAR) else kernel_poly if (k_t == POLY) else kernel_rbf if (k_t == RBF) else kernel_sigmoid if (k_t == SIGMOID) else kernel_precomputed`

## Parameters

- **k_t** (`Int`)

## Fields

- **l** (`Int`)
- **cache** (`Cache`)
- **sign** (`Optional[UnsafePointer[Int8, MutExternalOrigin]]`)
- **index** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)
- **next_buffer** (`Int`)
- **buffer** (`InlineArray[Optional[UnsafePointer[Float32, MutExternalOrigin]], 2]`)
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

### `swap_index`

```mojo
fn swap_index(self, i: Int, j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

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


