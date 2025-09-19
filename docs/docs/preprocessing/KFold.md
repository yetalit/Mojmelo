Mojo function

# `KFold`

```mojo
fn KFold[m_type: CVM](mut model: m_type, X: Matrix, y: Matrix, scoring: fn(Matrix, Matrix) raises -> SIMD[float32, 1], n_splits: Int = 5) -> SIMD[float32, 1]
```

K-Fold cross-validator.

**Parameters:**

- **m_type** (`CVM`): Model type.

**Args:**

- **model** (`m_type`): Model.
- **X** (`Matrix`): Samples.
- **y** (`Matrix`): Targets.
- **scoring** (`fn(Matrix, Matrix) raises -> SIMD[float32, 1]`): Scoring function.
- **n_splits** (`Int`): Number of folds.

**Returns:**

`SIMD`: Score.

**Raises:**

```mojo
fn KFold[m_type: CVP](mut model: m_type, X: Matrix, y: PythonObject, scoring: fn(PythonObject, List[String]) raises -> SIMD[float32, 1], n_splits: Int = 5) -> SIMD[float32, 1]
```

K-Fold cross-validator.

**Parameters:**

- **m_type** (`CVP`): Model type.

**Args:**

- **model** (`m_type`): Model.
- **X** (`Matrix`): Samples.
- **y** (`PythonObject`): Targets.
- **scoring** (`fn(PythonObject, List[String]) raises -> SIMD[float32, 1]`): Scoring function.
- **n_splits** (`Int`): Number of folds.

**Returns:**

`SIMD`: Score.

**Raises:**

