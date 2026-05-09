Mojo struct

# `head_t`

```mojo
@register_passable
struct head_t
```

## Fields

- **prev** (`Optional[UnsafePointer[head_t, MutAnyOrigin]]`)
- **next** (`Optional[UnsafePointer[head_t, MutAnyOrigin]]`)
- **data** (`Optional[UnsafePointer[Float32, MutExternalOrigin]]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`, `Movable`, `RegisterPassable`

## Methods

### `__init__`

```mojo
fn __init__() -> Self
```

**Returns:**

`Self`


