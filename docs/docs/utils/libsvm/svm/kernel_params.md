Mojo struct

# `kernel_params`

```mojo
@memory_only
struct kernel_params
```

## Aliases

- `__del__is_trivial = True`

## Fields

- **x** (`UnsafePointer[UnsafePointer[svm_node]]`)
- **x_square** (`UnsafePointer[Float64]`)
- **kernel_type** (`Int`)
- **degree** (`Int`)
- **gamma** (`Float64`)
- **coef0** (`Float64`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

