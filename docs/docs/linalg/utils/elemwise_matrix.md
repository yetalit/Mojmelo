Mojo function

# `elemwise_matrix`

```mojo
fn def elemwise_matrix[dtype: DType, width: Int, func: def[dtype: DType, width: Int](SIMD[dtype, width], SIMD[dtype, width]) thin -> SIMD[dtype, width]](dst: Pointer[Scalar[dtype], MutUntrackedOrigin], lhs: Pointer[Scalar[dtype], MutUntrackedOrigin], rhs: Pointer[Scalar[dtype], MutUntrackedOrigin], count: Int)
```

**Parameters:**

- **dtype** (`DType`)
- **width** (`Int`)
- **func** (`def[dtype: DType, width: Int](SIMD[dtype, width], SIMD[dtype, width]) thin -> SIMD[dtype, width]`)

**Args:**

- **dst** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`)
- **lhs** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`)
- **rhs** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`)
- **count** (`Int`)

