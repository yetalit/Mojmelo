Mojo struct

# `svm_problem`

```mojo
@memory_only
struct svm_problem
```

## Fields

- **l** (`Int`)
- **y** (`UnsafePointer[SIMD[float64, 1]]`)
- **x** (`UnsafePointer[UnsafePointer[svm_node]]`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

