Mojo function

# `apply_householder_left`

```mojo
fn def apply_householder_left(mut M: Mat, essential: Vec, tau: Float64)
```

M <- (I - tau * w w^T) * M, where w = [1, essential...] has M.rows() entries.

**Args:**

- **M** (`Mat`)
- **essential** (`Vec`)
- **tau** (`Float64`)

