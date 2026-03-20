Mojo function

# `cast`

```mojo
def cast[src: DType, des: DType, width: Int](data: UnsafePointer[Scalar[src], MutAnyOrigin], size: Int) -> UnsafePointer[Scalar[des], MutExternalOrigin]
```

**Parameters:**

- **src** (`DType`)
- **des** (`DType`)
- **width** (`Int`)

**Args:**

- **data** (`UnsafePointer`)
- **size** (`Int`)

**Returns:**

`UnsafePointer`

