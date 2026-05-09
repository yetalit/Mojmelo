Mojo struct

# `SVC`

```mojo
@memory_only
struct SVC[kernel: Int = 2]
```

Support Vector Classification.

## Aliases

- `MODEL_ID = 6`

## Parameters

- **kernel** (`Int`)

## Fields

- **C** (`Float64`): Regularization parameter. When C != 0, C-Support Vector Classification model will be used.
- **nu** (`Float64`): An upper bound on the fraction of margin errors and a lower bound of the fraction of support vectors. When nu != 0, Nu-Support Vector Classification model will be used.
- **degree** (`Int`): Degree of the polynomial kernel function ('poly').
- **gamma** (`Float64`): Kernel coefficient for 'rbf', 'poly' and 'sigmoid': if gamma='scale' (default) or -1 is passed then it uses 1 / (n_features * X.var()); if gamma='auto' or -0.1, it uses 1 / n_features; if custom float value, it must be non-negative.
- **coef0** (`Float64`): Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
- **cache_size** (`Float64`): Specify the size of the kernel cache (in MB).
- **tol** (`Float64`): Tolerance for stopping criterion.
- **shrinking** (`Bool`): Whether to use the shrinking heuristic.
- **probability** (`Bool`): Whether to enable probability estimates.

## Implemented traits

`AnyType`, `CV`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__(out self, gamma: String = "scale", C: Float64 = 0, nu: Float64 = 0, degree: Int = 2, coef0: Float64 = 0, cache_size: Float64 = Float64("200"), tol: Float64 = 0.001, shrinking: Bool = True, probability: Bool = False, random_state: Int = -1)
```

**Args:**

- **gamma** (`String`)
- **C** (`Float64`)
- **nu** (`Float64`)
- **degree** (`Int`)
- **coef0** (`Float64`)
- **cache_size** (`Float64`)
- **tol** (`Float64`)
- **shrinking** (`Bool`)
- **probability** (`Bool`)
- **random_state** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn __init__(out self, gamma: Float64, C: Float64 = 0, nu: Float64 = 0, degree: Int = 2, coef0: Float64 = 0, cache_size: Float64 = Float64("200"), tol: Float64 = 0.001, shrinking: Bool = True, probability: Bool = False, random_state: Int = -1)
```

**Args:**

- **gamma** (`Float64`)
- **C** (`Float64`)
- **nu** (`Float64`)
- **degree** (`Int`)
- **coef0** (`Float64`)
- **cache_size** (`Float64`)
- **tol** (`Float64`)
- **shrinking** (`Bool`)
- **probability** (`Bool`)
- **random_state** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict[String, String]`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__del__`

```mojo
fn __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: Matrix)
```

Fit the SVM model according to the given training data.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn predict(self, X: Matrix) -> Matrix
```

Perform classification on samples in X.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: The predicted classes.

**Raises:**

### `decision_function`

```mojo
fn decision_function(self, X: Matrix) -> List[List[Float64]]
```

Evaluate the decision function for the samples in X.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`List[List[Float64]]`: The decision values in a 2D List format.

### `save`

```mojo
fn save(self, path: String)
```

Save model data necessary for prediction to the specified path.

**Args:**

- **self** (`Self`)
- **path** (`String`)

**Raises:**

### `load`

```mojo
@staticmethod
fn load(path: String) -> Self
```

Load a saved model from the specified path for prediction.

**Args:**

- **path** (`String`)

**Returns:**

`Self`

**Raises:**

### `support_vectors`

```mojo
fn support_vectors(self) -> Matrix
```

Get support vectors.

**Args:**

- **self** (`Self`)

**Returns:**

`Matrix`

**Raises:**


