Mojo struct

# `kernel_params`

```mojo
@register_passable
struct kernel_params
```

## Fields

- **x** (`Pointer[Pointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]`)
- **x_square** (`Pointer[Float64, MutUntrackedOrigin]`)
- **kernel_type** (`Int`)
- **degree** (`Int`)
- **gamma** (`Float64`)
- **coef0** (`Float64`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`, `RegisterPassable`

