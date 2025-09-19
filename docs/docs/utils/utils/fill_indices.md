Mojo function

# `fill_indices`

```mojo
fn fill_indices(N: Int) -> UnsafePointer[SIMD[index, 1]]
```

Generates indices from 0 to N.

**Args:**

- **N** (`Int`)

**Returns:**

`UnsafePointer`: The pointer to indices.

**Raises:**

