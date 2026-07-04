Mojo function

# `cast`

```mojo
fn def cast[src: DType, des: DType, width: Int](data: UnsafePointer[Scalar[src], MutAnyOrigin], size: Int) -> UnsafePointer[Scalar[des], MutUntrackedOrigin]
```

**Parameters:**

- **src** (`DType`)
- **des** (`DType`)
- **width** (`Int`)

**Args:**

- **data** (`UnsafePointer[Scalar[src], MutAnyOrigin]`)
- **size** (`Int`)

**Returns:**

`UnsafePointer[Scalar[des], MutUntrackedOrigin]`

