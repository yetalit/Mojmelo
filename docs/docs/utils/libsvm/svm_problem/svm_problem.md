Mojo struct

# `svm_problem`

```mojo
@register_passable_trivial
struct svm_problem
```

## Aliases

- `__del__is_trivial = True`
- `__moveinit__is_trivial = True`
- `__copyinit__is_trivial = True`

## Fields

- **l** (`Int`)
- **y** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **x** (`UnsafePointer[UnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__() -> Self
```

**Returns:**

`Self`


