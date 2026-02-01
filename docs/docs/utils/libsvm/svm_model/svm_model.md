Mojo struct

# `svm_model`

```mojo
@memory_only
struct svm_model
```

## Aliases

- `__del__is_trivial = True`

## Fields

- **param** (`svm_parameter`)
- **nr_class** (`Int`)
- **l** (`Int`)
- **SV** (`UnsafePointer[UnsafePointer[svm_node, MutExternalOrigin], MutExternalOrigin]`)
- **sv_coef** (`UnsafePointer[UnsafePointer[Float64, MutExternalOrigin], MutExternalOrigin]`)
- **rho** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **probA** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **probB** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **prob_density_marks** (`UnsafePointer[Float64, MutExternalOrigin]`)
- **sv_indices** (`UnsafePointer[Scalar[DType.index], MutExternalOrigin]`)
- **label** (`UnsafePointer[Int, MutExternalOrigin]`)
- **nSV** (`UnsafePointer[Int, MutExternalOrigin]`)
- **free_sv** (`Int`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

