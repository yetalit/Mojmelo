Mojo struct

# `Kernel`

```mojo
@memory_only
struct Kernel
```

## Fields

- **x** (`UnsafePointer[UnsafePointer[svm_node]]`)
- **x_square** (`UnsafePointer[SIMD[float64, 1]]`)
- **kernel_type** (`Int`)
- **degree** (`Int`)
- **gamma** (`SIMD[float64, 1]`)
- **coef0** (`SIMD[float64, 1]`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

