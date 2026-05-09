Mojo struct

# `svm_problem`

```mojo
@register_passable
struct svm_problem
```

## Fields

- **l** (`Int`)
- **y** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **x** (`Optional[UnsafePointer[Optional[UnsafePointer[svm_node, MutExternalOrigin]], MutExternalOrigin]]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`, `Movable`, `RegisterPassable`

## Methods

### `__init__`

```mojo
fn __init__() -> Self
```

**Returns:**

`Self`


