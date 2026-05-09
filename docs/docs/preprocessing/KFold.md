Mojo function

# `KFold`

```mojo
fn KFold[m_type: CV, scoring: String](mut model: m_type, X: Matrix, y: Matrix, n_splits: Int = 5) -> Float32
```

K-Fold cross-validator.

**Parameters:**

- **m_type** (`CV`): Model type.
- **scoring** (`String`): The scoring function:
    accuracy_score -> 'accuracy';
    r2_score -> 'r2';
    MSE -> 'mse'.

**Args:**

- **model** (`m_type`): Model.
- **X** (`Matrix`): Samples.
- **y** (`Matrix`): Targets.
- **n_splits** (`Int`): Number of folds.

**Returns:**

`Float32`: Score.

**Raises:**

