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
- **SV** (`Optional[UnsafePointer[Optional[UnsafePointer[svm_node, MutExternalOrigin]], MutExternalOrigin]]`)
- **sv_coef** (`Optional[UnsafePointer[Optional[UnsafePointer[Float64, MutExternalOrigin]], MutExternalOrigin]]`)
- **rho** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **probA** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **probB** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **prob_density_marks** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **sv_indices** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)
- **label** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)
- **nSV** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)
- **free_sv** (`Int`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

