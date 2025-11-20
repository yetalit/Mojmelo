Mojo function

# `cast`

```mojo
fn cast[src: DType, des: DType, width: Int](data: UnsafePointer[Scalar[src], MutAnyOrigin], size: Int) -> UnsafePointer[Scalar[des], origin_of(MutOrigin.external)]
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

