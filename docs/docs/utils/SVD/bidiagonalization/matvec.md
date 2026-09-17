Mojo function

# `matvec`

```mojo
fn def matvec(A: Mat, x: Vec) -> Vec
```

Y = A * X, via column-scaled accumulation (each column read once, contiguous). y has length A.rows(), x must have length A.cols().

**Args:**

- **A** (`Mat`)
- **x** (`Vec`)

**Returns:**

`Vec`

