Mojo function

# `apply_rotation_right_cols`

```mojo
fn def apply_rotation_right_cols(mut M: Mat, p: Int, q: Int, rot: JacobiRotation)
```

M.cols[{p,q}] <- M.cols[{p,q}] * rot  (Eigen's applyOnTheRight).

**Args:**

- **M** (`Mat`)
- **p** (`Int`)
- **q** (`Int`)
- **rot** (`JacobiRotation`)

