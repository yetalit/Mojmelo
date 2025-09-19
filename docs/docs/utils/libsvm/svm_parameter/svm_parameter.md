Mojo struct

# `svm_parameter`

```mojo
@memory_only
struct svm_parameter
```

## Aliases

- `C_SVC = 0`
- `NU_SVC = 1`
- `ONE_CLASS = 2`
- `EPSILON_SVR = 3`
- `NU_SVR = 4`
- `LINEAR = 0`
- `POLY = 1`
- `RBF = 2`
- `SIGMOID = 3`
- `PRECOMPUTED = 4`

## Fields

- **svm_type** (`Int`)
- **kernel_type** (`Int`)
- **degree** (`Int`)
- **gamma** (`SIMD[float64, 1]`)
- **coef0** (`SIMD[float64, 1]`)
- **cache_size** (`SIMD[float64, 1]`)
- **eps** (`SIMD[float64, 1]`)
- **C** (`SIMD[float64, 1]`)
- **nr_weight** (`Int`)
- **weight_label** (`UnsafePointer[Int]`)
- **weight** (`UnsafePointer[SIMD[float64, 1]]`)
- **nu** (`SIMD[float64, 1]`)
- **p** (`SIMD[float64, 1]`)
- **shrinking** (`Int`)
- **probability** (`Int`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

