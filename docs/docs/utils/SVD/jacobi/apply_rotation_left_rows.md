Mojo function

# `apply_rotation_left_rows`

```mojo
fn def apply_rotation_left_rows(mut M: Mat, p: Int, q: Int, rot: JacobiRotation)
```

M.rows[{p,q}] <- rot * M.rows[{p,q}]  (Eigen's applyOnTheLeft).

**Args:**

- **M** (`Mat`)
- **p** (`Int`)
- **q** (`Int`)
- **rot** (`JacobiRotation`)

