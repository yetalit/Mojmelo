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
- **SV** (`Optional[UnsafePointer[UnsafePointer[svm_node, MutUntrackedOrigin], MutUntrackedOrigin]]`)
- **sv_coef** (`Optional[UnsafePointer[Optional[UnsafePointer[Float64, MutUntrackedOrigin]], MutUntrackedOrigin]]`)
- **rho** (`Optional[UnsafePointer[Float64, MutUntrackedOrigin]]`)
- **probA** (`Optional[UnsafePointer[Float64, MutUntrackedOrigin]]`)
- **probB** (`Optional[UnsafePointer[Float64, MutUntrackedOrigin]]`)
- **prob_density_marks** (`Optional[UnsafePointer[Float64, MutUntrackedOrigin]]`)
- **sv_indices** (`Optional[UnsafePointer[Int, MutUntrackedOrigin]]`)
- **label** (`Optional[UnsafePointer[Int, MutUntrackedOrigin]]`)
- **nSV** (`Optional[UnsafePointer[Int, MutUntrackedOrigin]]`)
- **free_sv** (`Int`)

## Implemented traits

`AnyType`, `ImplicitlyDeletable`

