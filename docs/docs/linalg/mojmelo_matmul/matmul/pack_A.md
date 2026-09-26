Mojo function

# `pack_A`

```mojo
fn def pack_A[Type: DType, //, mr: Int, inner_parallel: Bool = False](Ac_buffer: Pointer[Scalar[Type], MutUntrackedOrigin], Ac: Matrix[Type]) -> Matrix[Type]
```

**Parameters:**

- **Type** (`DType`)
- **mr** (`Int`)
- **inner_parallel** (`Bool`)

**Args:**

- **Ac_buffer** (`Pointer[Scalar[Type], MutUntrackedOrigin]`)
- **Ac** (`Matrix[Type]`)

**Returns:**

`Matrix[Type]`

