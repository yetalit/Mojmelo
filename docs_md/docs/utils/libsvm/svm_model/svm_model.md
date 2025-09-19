Mojo struct

# `svm_model`

```mojo
@memory_only
struct svm_model
```

## Fields

- **param** (`svm_parameter`)
- **nr_class** (`Int`)
- **l** (`Int`)
- **SV** (`UnsafePointer[UnsafePointer[svm_node]]`)
- **sv_coef** (`UnsafePointer[UnsafePointer[SIMD[float64, 1]]]`)
- **rho** (`UnsafePointer[SIMD[float64, 1]]`)
- **probA** (`UnsafePointer[SIMD[float64, 1]]`)
- **probB** (`UnsafePointer[SIMD[float64, 1]]`)
- **prob_density_marks** (`UnsafePointer[SIMD[float64, 1]]`)
- **sv_indices** (`UnsafePointer[Int]`)
- **label** (`UnsafePointer[Int]`)
- **nSV** (`UnsafePointer[Int]`)
- **free_sv** (`Int`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

