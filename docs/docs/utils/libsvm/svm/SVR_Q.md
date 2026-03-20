Mojo struct

# `SVR_Q`

```mojo
@memory_only
struct SVR_Q
```

## Fields

- **l** (`Int`)
- **cache** (`Cache`)
- **sign** (`UnsafePointer[Int8, MutExternalOrigin]`)
- **index** (`UnsafePointer[Int, MutExternalOrigin]`)
- **next_buffer** (`Int`)
- **buffer** (`InlineArray[UnsafePointer[Float32, MutExternalOrigin], 2]`)
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

### `swap_index`

```mojo
def swap_index(self, i: Int, j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

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


