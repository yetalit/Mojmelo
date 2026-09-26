Mojo struct

# `svm_parameter`

```mojo
@memory_only
struct svm_parameter
```

## Aliases

- `C_SVC = Int(0)`
- `NU_SVC = Int(1)`
- `ONE_CLASS = Int(2)`
- `EPSILON_SVR = Int(3)`
- `NU_SVR = Int(4)`
- `LINEAR = Int(0)`
- `POLY = Int(1)`
- `RBF = Int(2)`
- `SIGMOID = Int(3)`
- `PRECOMPUTED = Int(4)`

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
- **weight_label** (`Optional[Pointer[Int, MutUntrackedOrigin]]`)
- **weight** (`Optional[Pointer[Float64, MutUntrackedOrigin]]`)
- **nu** (`Float64`)
- **p** (`Float64`)
- **shrinking** (`Int`)
- **probability** (`Int`)

## Implemented traits

`AnyType`, `Copyable`, `Deinitable`, `Movable`

