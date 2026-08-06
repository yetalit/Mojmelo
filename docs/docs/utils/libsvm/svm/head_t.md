Mojo struct

# `head_t`

```mojo
@register_passable
struct head_t
```

## Fields

- **prev** (`Optional[Pointer[head_t, MutUntrackedOrigin]]`)
- **next** (`Optional[Pointer[head_t, MutUntrackedOrigin]]`)
- **data** (`Optional[Pointer[Float32, MutUntrackedOrigin]]`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`, `RegisterPassable`

## Methods

### `__init__`

```mojo
fn def __init__() -> Self
```

**Returns:**

`Self`


