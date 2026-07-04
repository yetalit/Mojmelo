Mojo struct

# `SVC_Q`

```mojo
@memory_only
struct SVC_Q
```

## Fields

- **y** (`UnsafePointer[Int8, MutUntrackedOrigin]`)
- **cache** (`Cache`)
- **QD** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **kernel_function** (`def(kernel_params, Int, Int) -> Float64`)

## Implemented traits

`AnyType`, `ImplicitlyDeletable`, `QMatrix`

## Methods

### `__init__`

```mojo
fn def __init__(out self, prob: svm_problem, param: svm_parameter, y_: Optional[UnsafePointer[Int8, MutUntrackedOrigin]])
```

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **y_** (`Optional[UnsafePointer[Int8, MutUntrackedOrigin]]`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

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

### `swap_index`

```mojo
fn def swap_index(mut self, i: Int, j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)


