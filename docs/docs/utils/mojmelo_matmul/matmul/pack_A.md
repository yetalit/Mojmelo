Mojo function

# `pack_A`

```mojo
fn pack_A[Type: DType, //, mr: Int, inner_parallel: Bool = False](Ac_buffer: UnsafePointer[Scalar[Type], MutAnyOrigin], Ac: Matrix[Type]) -> Matrix[Type]
```

**Parameters:**

- **Type** (`DType`)
- **mr** (`Int`)
- **inner_parallel** (`Bool`)

**Args:**

- **Ac_buffer** (`UnsafePointer[Scalar[Type], MutAnyOrigin]`)
- **Ac** (`Matrix[Type]`)

**Returns:**

`Matrix[Type]`

