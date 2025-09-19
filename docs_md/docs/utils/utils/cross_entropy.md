Mojo function

# `cross_entropy`

```mojo
fn cross_entropy(y: Matrix, y_pred: Matrix) -> SIMD[float32, 1]
```

Binary Cross Entropy.

**Args:**

- **y** (`Matrix`)
- **y_pred** (`Matrix`)

**Returns:**

`SIMD`: The loss.

**Raises:**

