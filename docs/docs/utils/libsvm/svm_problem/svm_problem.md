Mojo struct

# `svm_problem`

```mojo
@register_passable_trivial
struct svm_problem
```

## Fields

- **l** (`Int`)
- **y** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **x** (`UnsafePointer[UnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Movable`, `RegisterPassable`, `TrivialRegisterPassable`

## Methods

### `__init__`

```mojo
def __init__() -> Self
```

**Returns:**

`Self`


