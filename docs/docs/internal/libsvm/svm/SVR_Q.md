Mojo struct

# `SVR_Q`

```mojo
@memory_only
struct SVR_Q
```

## Fields

- **l** (`Int`)
- **cache** (`Cache`)
- **sign** (`Pointer[Int8, MutUntrackedOrigin]`)
- **index** (`Pointer[Int, MutUntrackedOrigin]`)
- **next_buffer** (`Int`)
- **buffer** (`Array[Optional[Pointer[Float32, MutUntrackedOrigin]], Int(2)]`)
- **QD** (`Pointer[Float64, MutUntrackedOrigin]`)
- **kernel_function** (`def(kernel_params, Int, Int) thin -> Float64`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`, `QMatrix`

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

### `__deinit__`

```mojo
fn def __deinit__(deinit self)
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
fn def get_Q(mut self, i: Int, _len: Int) -> Pointer[Float32, MutUntrackedOrigin]
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **_len** (`Int`)

**Returns:**

`Pointer[Float32, MutUntrackedOrigin]`

### `get_QD`

```mojo
fn def get_QD(self) -> Pointer[Float64, MutUntrackedOrigin]
```

**Args:**

- **self** (`Self`)

**Returns:**

`Pointer[Float64, MutUntrackedOrigin]`


