Mojo struct

# `SVR_Q`

```mojo
@memory_only
struct SVR_Q
```

## Aliases

- `__del__is_trivial = False`

## Fields

- **l** (`Int`)
- **cache** (`Cache`)
- **sign** (`UnsafePointer[Int8]`)
- **index** (`UnsafePointer[Int]`)
- **next_buffer** (`Int`)
- **buffer** (`InlineArray[UnsafePointer[Float32], 2]`)
- **QD** (`UnsafePointer[Float64]`)
- **kernel_function** (`fn(kernel_params, Int, Int) -> Float64`)

## Implemented traits

`AnyType`, `QMatrix`, `UnknownDestructibility`

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
fn __del__(var self)
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
fn get_Q(mut self, i: Int, _len: Int) -> UnsafePointer[Float32]
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **_len** (`Int`)

**Returns:**

`UnsafePointer`

### `get_QD`

```mojo
fn get_QD(self) -> UnsafePointer[Float64]
```

**Args:**

- **self** (`Self`)

**Returns:**

`UnsafePointer`


