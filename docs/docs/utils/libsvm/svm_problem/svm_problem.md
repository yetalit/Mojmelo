Mojo struct

# `svm_problem`

```mojo
@register_passable
struct svm_problem
```

## Fields

- **l** (`Int`)
- **y** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **x** (`UnsafePointer[UnsafePointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]`)

## Implemented traits

`AnyType`, `ImplicitlyDeletable`, `Movable`, `RegisterPassable`

## Methods

### `__init__`

```mojo
fn def __init__() -> Self
```

**Returns:**

`Self`


