Mojo function

# `matvec`

```mojo
fn def matvec(A: Mat, x: Vec) -> Vec
```

Y = A * x. Each task owns a slice of rows of y (no write sharing) and consumes 4 columns per pass, so y is loaded/stored once per 4 FMAs.

**Args:**

- **A** (`Mat`)
- **x** (`Vec`)

**Returns:**

`Vec`

