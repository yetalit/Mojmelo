Mojo function

# `svd_thin`

```mojo
fn def svd_thin(m: Int, n: Int, k: Int, S: UnsafePointer[Float64, MutUntrackedOrigin], mut Vout: Matrix, ATA: UnsafePointer[Float64, MutAnyOrigin])
```

**Args:**

- **m** (`Int`)
- **n** (`Int`)
- **k** (`Int`)
- **S** (`UnsafePointer[Float64, MutUntrackedOrigin]`)
- **Vout** (`Matrix`)
- **ATA** (`UnsafePointer[Float64, MutAnyOrigin]`)

**Raises:**

