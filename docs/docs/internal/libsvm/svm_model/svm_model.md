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
- **SV** (`Optional[Pointer[Pointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]]`)
- **sv_coef** (`Optional[Pointer[Optional[Pointer[Float64, MutUntrackedOrigin]], MutUntrackedOrigin]]`)
- **rho** (`Optional[Pointer[Float64, MutUntrackedOrigin]]`)
- **probA** (`Optional[Pointer[Float64, MutUntrackedOrigin]]`)
- **probB** (`Optional[Pointer[Float64, MutUntrackedOrigin]]`)
- **prob_density_marks** (`Optional[Pointer[Float64, MutUntrackedOrigin]]`)
- **sv_indices** (`Optional[Pointer[Int, MutUntrackedOrigin]]`)
- **label** (`Optional[Pointer[Int, MutUntrackedOrigin]]`)
- **nSV** (`Optional[Pointer[Int, MutUntrackedOrigin]]`)
- **free_sv** (`Int`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`

