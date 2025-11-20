Mojo struct

# `head_t`

```mojo
@register_passable_trivial
struct head_t
```

## Aliases

- `__del__is_trivial = True`
- `__moveinit__is_trivial = True`
- `__copyinit__is_trivial = True`

## Fields

- **prev** (`UnsafePointer[head_t, MutAnyOrigin]`)
- **next** (`UnsafePointer[head_t, MutAnyOrigin]`)
- **data** (`UnsafePointer[Float32, origin_of(MutOrigin.external)]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__() -> Self
```

**Returns:**

`Self`


