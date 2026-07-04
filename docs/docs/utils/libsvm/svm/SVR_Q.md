Mojo struct

# `SVR_Q`

```mojo
@memory_only
struct SVR_Q
```

## Fields

- **l** (`Int`)
- **cache** (`Cache`)
- **sign** (`UnsafePointer[Int8, MutUntrackedOrigin]`)
- **index** (`UnsafePointer[Int, MutUntrackedOrigin]`)
- **next_buffer** (`Int`)
- **buffer** (`InlineArray[Optional[UnsafePointer[Float32, MutUntrackedOrigin]], 2]`)
- **QD** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **kernel_function** (`def(kernel_params, Int, Int) -> Float64`)

## Implemented traits

`AnyType`, `ImplicitlyDeletable`, `QMatrix`

## Methods

### `__init__`

```mojo
fn def __init__(out self, prob: svm_problem, param: svm_parameter)
```

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `swap_index`

```mojo
fn def swap_index(self, i: Int, j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

### `get_Q`

```mojo
fn def get_Q(mut self, i: Int, _len: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **_len** (`Int`)

**Returns:**

`UnsafePointer[Float32, MutUntrackedOrigin]`

### `get_QD`

```mojo
fn def get_QD(self) -> UnsafePointer[Float64, MutUntrackedOrigin]
```

**Args:**

- **self** (`Self`)

**Returns:**

`UnsafePointer[Float64, MutUntrackedOrigin]`


