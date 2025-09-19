Mojo function

# `normalize`

```mojo
fn normalize(data: Matrix, norm: String = "l2") -> Tuple[Matrix, Matrix]
```

Scale input vectors individually to unit norm (vector length).

**Args:**

- **data** (`Matrix`): Data.
- **norm** (`String`): The norm to use -> 'l2', 'l1'.

**Returns:**

`Tuple`: Normalized data, norms.

**Raises:**

