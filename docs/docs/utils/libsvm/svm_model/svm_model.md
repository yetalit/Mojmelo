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
- **SV** (`UnsafePointer[UnsafePointer[svm_node]]`)
- **sv_coef** (`UnsafePointer[UnsafePointer[Float64]]`)
- **rho** (`UnsafePointer[Float64]`)
- **probA** (`UnsafePointer[Float64]`)
- **probB** (`UnsafePointer[Float64]`)
- **prob_density_marks** (`UnsafePointer[Float64]`)
- **sv_indices** (`UnsafePointer[Scalar[DType.index]]`)
- **label** (`UnsafePointer[Int]`)
- **nSV** (`UnsafePointer[Int]`)
- **free_sv** (`Int`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

