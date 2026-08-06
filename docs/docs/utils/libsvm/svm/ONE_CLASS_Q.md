Mojo struct

# `ONE_CLASS_Q`

```mojo
@memory_only
struct ONE_CLASS_Q
```

## Fields

- **cache** (`Cache`)
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

### `swap_index`

```mojo
fn def swap_index(mut self, i: Int, j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)


