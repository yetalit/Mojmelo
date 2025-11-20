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
- **SV** (`UnsafePointer[UnsafePointer[svm_node, origin_of(MutOrigin.external)], origin_of(MutOrigin.external)]`)
- **sv_coef** (`UnsafePointer[UnsafePointer[Float64, origin_of(MutOrigin.external)], origin_of(MutOrigin.external)]`)
- **rho** (`UnsafePointer[Float64, origin_of(MutOrigin.external)]`)
- **probA** (`UnsafePointer[Float64, origin_of(MutOrigin.external)]`)
- **probB** (`UnsafePointer[Float64, origin_of(MutOrigin.external)]`)
- **prob_density_marks** (`UnsafePointer[Float64, origin_of(MutOrigin.external)]`)
- **sv_indices** (`UnsafePointer[Scalar[DType.int], origin_of(MutOrigin.external)]`)
- **label** (`UnsafePointer[Int, origin_of(MutOrigin.external)]`)
- **nSV** (`UnsafePointer[Int, origin_of(MutOrigin.external)]`)
- **free_sv** (`Int`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

