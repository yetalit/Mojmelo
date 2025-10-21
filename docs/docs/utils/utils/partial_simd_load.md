Mojo function

# `partial_simd_load`

```mojo
fn partial_simd_load[width: Int](data: UnsafePointer[Float32], offset: Int, size: Int) -> SIMD[DType.float32, width]
```

**Parameters:**

- **width** (`Int`)

**Args:**

- **data** (`UnsafePointer`)
- **offset** (`Int`)
- **size** (`Int`)

**Returns:**

`SIMD`

