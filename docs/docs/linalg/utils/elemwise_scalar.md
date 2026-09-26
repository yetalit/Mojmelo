Mojo function

# `elemwise_scalar`

```mojo
fn def elemwise_scalar[dtype: DType, width: Int, func: def[dtype: DType, width: Int](SIMD[dtype, width], SIMD[dtype, width]) thin -> SIMD[dtype, width]](dst: Pointer[Scalar[dtype], MutUntrackedOrigin], src: Pointer[Scalar[dtype], MutUntrackedOrigin], count: Int, s: Scalar[dtype])
```

**Parameters:**

- **dtype** (`DType`)
- **width** (`Int`)
- **func** (`def[dtype: DType, width: Int](SIMD[dtype, width], SIMD[dtype, width]) thin -> SIMD[dtype, width]`)

**Args:**

- **dst** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`)
- **src** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`)
- **count** (`Int`)
- **s** (`Scalar[dtype]`)

