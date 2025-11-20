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
- **y** (`UnsafePointer[Float64, origin_of(MutOrigin.external)]`)
- **x** (`UnsafePointer[UnsafePointer[svm_node, origin_of(MutOrigin.external)], origin_of(MutOrigin.external)]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__() -> Self
```

**Returns:**

`Self`


