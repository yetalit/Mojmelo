Mojo struct

# `kernel_params`

```mojo
@register_passable
struct kernel_params
```

## Fields

- **x** (`Optional[UnsafePointer[Optional[UnsafePointer[svm_node, MutExternalOrigin]], MutExternalOrigin]]`)
- **x_square** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **kernel_type** (`Int`)
- **degree** (`Int`)
- **gamma** (`Float64`)
- **coef0** (`Float64`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`, `Movable`, `RegisterPassable`

