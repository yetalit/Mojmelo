Mojo struct

# `Kernel`

```mojo
@memory_only
struct Kernel
```

## Aliases

- `__del__is_trivial = False`

## Fields

- **kernel_function** (`fn(kernel_params, Int, Int) -> Float64`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, l: Int, x_: UnsafePointer[UnsafePointer[svm_node, origin_of(MutOrigin.external)], origin_of(MutOrigin.external)], param: svm_parameter)
```

**Args:**

- **l** (`Int`)
- **x_** (`UnsafePointer`)
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


