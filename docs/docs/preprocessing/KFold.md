Mojo function

# `KFold`

```mojo
fn KFold[m_type: CV](mut model: m_type, X: Matrix, y: Matrix, scoring: fn(Matrix, Matrix) raises -> Float32, n_splits: Int = 5) -> Float32
```

K-Fold cross-validator.

**Parameters:**

- **m_type** (`CV`): Model type.

**Args:**

- **model** (`m_type`): Model.
- **X** (`Matrix`): Samples.
- **y** (`Matrix`): Targets.
- **scoring** (`fn(Matrix, Matrix) raises -> Float32`): Scoring function.
- **n_splits** (`Int`): Number of folds.

**Returns:**

`Float32`: Score.

**Raises:**

