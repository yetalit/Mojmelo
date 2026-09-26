Mojo struct

# `svm_problem`

```mojo
@register_passable
struct svm_problem
```

## Fields

- **l** (`Int`)
- **y** (`Pointer[Float64, MutUntrackedOrigin]`)
- **x** (`Pointer[Pointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`, `RegisterPassable`

## Methods

### `__init__`

```mojo
fn def __init__() -> Self
```

**Returns:**

`Self`


