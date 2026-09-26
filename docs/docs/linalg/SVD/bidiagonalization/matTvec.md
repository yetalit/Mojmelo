Mojo function

# `matTvec`

```mojo
fn def matTvec(A: Mat, x: Vec) -> Vec
```

Y = A^T x. One dot per column. Tasks own *groups* of columns sized to ~32K elements so dispatch cost is amortised.

**Args:**

- **A** (`Mat`)
- **x** (`Vec`)

**Returns:**

`Vec`

