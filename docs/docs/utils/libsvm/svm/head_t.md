Mojo struct

# `head_t`

```mojo
@register_passable
struct head_t
```

## Fields

- **prev** (`Optional[UnsafePointer[head_t, MutAnyOrigin]]`)
- **next** (`Optional[UnsafePointer[head_t, MutAnyOrigin]]`)
- **data** (`Optional[UnsafePointer[Float32, MutUntrackedOrigin]]`)

## Implemented traits

`AnyType`, `ImplicitlyDeletable`, `Movable`, `RegisterPassable`

## Methods

### `__init__`

```mojo
fn def __init__() -> Self
```

**Returns:**

`Self`


