Mojo struct

# `kernel_params`

```mojo
@register_passable_trivial
struct kernel_params
```

## Fields

- **x** (`UnsafePointer[UnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]`)
- **x_square** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **kernel_type** (`Int`)
- **degree** (`Int`)
- **gamma** (`Float64`)
- **coef0** (`Float64`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Movable`, `RegisterPassable`, `TrivialRegisterPassable`

