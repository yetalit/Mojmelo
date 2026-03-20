Mojo struct

# `ONE_CLASS_Q`

```mojo
@memory_only
struct ONE_CLASS_Q
```

## Fields

- **cache** (`Cache`)
- **QD** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **kernel_function** (`def(kernel_params, Int, Int) -> Float64`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`, `QMatrix`

## Methods

### `__init__`

```mojo
def __init__(out self, prob: svm_problem, param: svm_parameter)
```

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `get_Q`

```mojo
def get_Q(mut self, i: Int, _len: Int) -> UnsafePointer[Float32, MutExternalOrigin]
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **_len** (`Int`)

**Returns:**

`UnsafePointer`

### `get_QD`

```mojo
def get_QD(self) -> UnsafePointer[Float64, MutExternalOrigin]
```

**Args:**

- **self** (`Self`)

**Returns:**

`UnsafePointer`

### `swap_index`

```mojo
def swap_index(mut self, i: Int, j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)


