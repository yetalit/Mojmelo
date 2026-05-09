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
- **weight_label** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)
- **weight** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **nu** (`Float64`)
- **p** (`Float64`)
- **shrinking** (`Int`)
- **probability** (`Int`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDestructible`, `Movable`

