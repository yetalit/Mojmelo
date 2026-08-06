Mojo function

# `cast`

```mojo
fn def cast[src: DType, des: DType, width: Int](data: Pointer[Scalar[src], MutUntrackedOrigin], size: Int) -> Pointer[Scalar[des], MutUntrackedOrigin]
```

**Parameters:**

- **src** (`DType`)
- **des** (`DType`)
- **width** (`Int`)

**Args:**

- **data** (`Pointer[Scalar[src], MutUntrackedOrigin]`)
- **size** (`Int`)

**Returns:**

`Pointer[Scalar[des], MutUntrackedOrigin]`

