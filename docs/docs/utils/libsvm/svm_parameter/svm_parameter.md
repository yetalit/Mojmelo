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
- `__del__is_trivial = True`
- `__moveinit__is_trivial = True`
- `__copyinit__is_trivial = True`

## Fields

- **svm_type** (`Int`)
- **kernel_type** (`Int`)
- **degree** (`Int`)
- **gamma** (`Float64`)
- **coef0** (`Float64`)
- **cache_size** (`Float64`)
- **eps** (`Float64`)
- **C** (`Float64`)
- **nr_weight** (`Int`)
- **weight_label** (`UnsafePointer[Int]`)
- **weight** (`UnsafePointer[Float64]`)
- **nu** (`Float64`)
- **p** (`Float64`)
- **shrinking** (`Int`)
- **probability** (`Int`)

## Implemented traits

`AnyType`, `Copyable`, `Movable`, `UnknownDestructibility`

