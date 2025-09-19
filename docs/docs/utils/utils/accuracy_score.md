Mojo function

# `accuracy_score`

```mojo
fn accuracy_score(y: Matrix, y_pred: Matrix) -> SIMD[float32, 1]
```

Accuracy classification score.

**Args:**

- **y** (`Matrix`)
- **y_pred** (`Matrix`)

**Returns:**

`SIMD`: The score.

**Raises:**

```mojo
fn accuracy_score(y: List[String], y_pred: List[String]) -> SIMD[float32, 1]
```

Accuracy classification score.

**Args:**

- **y** (`List`)
- **y_pred** (`List`)

**Returns:**

`SIMD`: The score.

**Raises:**

```mojo
fn accuracy_score(y: PythonObject, y_pred: Matrix) -> SIMD[float32, 1]
```

Accuracy classification score.

**Args:**

- **y** (`PythonObject`)
- **y_pred** (`Matrix`)

**Returns:**

`SIMD`: The score.

**Raises:**

```mojo
fn accuracy_score(y: PythonObject, y_pred: List[String]) -> SIMD[float32, 1]
```

Accuracy classification score.

**Args:**

- **y** (`PythonObject`)
- **y_pred** (`List`)

**Returns:**

`SIMD`: The score.

**Raises:**

