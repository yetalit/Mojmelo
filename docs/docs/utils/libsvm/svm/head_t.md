Mojo struct

# `head_t`

```mojo
@register_passable_trivial
struct head_t
```

## Fields

- **prev** (`UnsafePointer[head_t, MutAnyOrigin]`)
- **next** (`UnsafePointer[head_t, MutAnyOrigin]`)
- **data** (`UnsafePointer[Float32, MutExternalOrigin]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Movable`, `RegisterPassable`, `TrivialRegisterPassable`

## Methods

### `__init__`

```mojo
def __init__() -> Self
```

**Returns:**

`Self`


