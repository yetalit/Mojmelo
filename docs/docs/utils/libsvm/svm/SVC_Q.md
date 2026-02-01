Mojo struct

# `SVC_Q`

```mojo
@memory_only
struct SVC_Q
```

## Aliases

- `__del__is_trivial = False`

## Fields

- **y** (`UnsafePointer[Int8, MutExternalOrigin]`)
- **cache** (`Cache`)
- **QD** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **kernel_function** (`fn(kernel_params, Int, Int) -> Float64`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`, `QMatrix`

## Methods

### `__init__`

```mojo
fn __init__(out self, prob: svm_problem, param: svm_parameter, y_: UnsafePointer[Int8, MutExternalOrigin])
```

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **y_** (`UnsafePointer`)
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
fn get_Q(mut self, i: Int, _len: Int) -> UnsafePointer[Float32, MutExternalOrigin]
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **_len** (`Int`)

**Returns:**

`UnsafePointer`

### `get_QD`

```mojo
fn get_QD(self) -> UnsafePointer[Float64, MutExternalOrigin]
```

**Args:**

- **self** (`Self`)

**Returns:**

`UnsafePointer`

### `swap_index`

```mojo
fn swap_index(mut self, i: Int, j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)


