Mojo function

# `apply_householder_right`

```mojo
fn def apply_householder_right(mut M: Mat, essential: Vec, tau: Float64)
```

M <- M * (I - tau * w w^T), where w = [1, essential...] has M.cols() entries.

**Args:**

- **M** (`Mat`)
- **essential** (`Vec`)
- **tau** (`Float64`)

