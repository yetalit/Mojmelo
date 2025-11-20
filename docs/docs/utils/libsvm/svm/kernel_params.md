Mojo struct

# `kernel_params`

```mojo
@register_passable_trivial
struct kernel_params
```

## Aliases

- `__del__is_trivial = True`
- `__moveinit__is_trivial = True`
- `__copyinit__is_trivial = True`

## Fields

- **x** (`UnsafePointer[UnsafePointer[svm_node, origin_of(MutOrigin.external)], origin_of(MutOrigin.external)]`)
- **x_square** (`UnsafePointer[Float64, origin_of(MutOrigin.external)]`)
- **kernel_type** (`Int`)
- **degree** (`Int`)
- **gamma** (`Float64`)
- **coef0** (`Float64`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `Movable`, `UnknownDestructibility`

