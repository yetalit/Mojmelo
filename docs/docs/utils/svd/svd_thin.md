Mojo function

# `svd_thin`

```mojo
fn svd_thin(m: Int, n: Int, k: Int, S: UnsafePointer[Float64, MutAnyOrigin], mut Vout: Matrix, ATA: UnsafePointer[Float64, MutAnyOrigin])
```

**Args:**

- **m** (`Int`)
- **n** (`Int`)
- **k** (`Int`)
- **S** (`UnsafePointer[Float64, MutAnyOrigin]`)
- **Vout** (`Matrix`)
- **ATA** (`UnsafePointer[Float64, MutAnyOrigin]`)

**Raises:**

