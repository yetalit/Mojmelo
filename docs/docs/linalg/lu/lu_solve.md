Mojo function

# `lu_solve`

```mojo
fn def lu_solve[dtype: DType = .float64](a: Pointer[Scalar[dtype], MutUntrackedOrigin], x: Pointer[Scalar[dtype], MutUntrackedOrigin], n: Int, nrhs: Int = Int(1))
```

Solve A X = B in place.

**Parameters:**

- **dtype** (`DType`)

**Args:**

- **a** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`): (n x n) matrix, row-major. Overwritten by the solve.
- **x** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`): (n x nrhs) right-hand sides, row-major (a plain length-n vector when
   nrhs == 1). Holds B on entry and the solution X on exit.
- **n** (`Int`): Matrix dimension.
- **nrhs** (`Int`): Number of right-hand-side columns.

**Raises:**

If n or nrhs is not positive, or A is singular to working precision.

