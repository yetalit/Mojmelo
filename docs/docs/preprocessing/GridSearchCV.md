Mojo function

# `GridSearchCV`

```mojo
fn GridSearchCV[m_type: CVM](X: Matrix, y: Matrix, param_grid: Dict[String, List[String]], scoring: fn(Matrix, Matrix) raises -> SIMD[float32, 1], neg_score: Bool = False, n_jobs: Int = 0, cv: Int = 5) -> Tuple[Dict[String, String], SIMD[float32, 1]]
```

Exhaustive search over specified parameter values for an estimator.

**Parameters:**

- **m_type** (`CVM`): Model type.

**Args:**

- **X** (`Matrix`): Samples.
- **y** (`Matrix`): Targets.
- **param_grid** (`Dict`): Dictionary with parameters names as keys and lists of parameter settings to try as values.
- **scoring** (`fn(Matrix, Matrix) raises -> SIMD[float32, 1]`): Scoring function.
- **neg_score** (`Bool`): Invert the scoring results when finding the best params.
- **n_jobs** (`Int`): Number of jobs to run in parallel. `-1` means using all processors.
- **cv** (`Int`): Number of folds in a KFold.

**Returns:**

`Tuple`: Best parameters.

**Raises:**

```mojo
fn GridSearchCV[m_type: CVP](X: Matrix, y: PythonObject, param_grid: Dict[String, List[String]], scoring: fn(PythonObject, List[String]) raises -> SIMD[float32, 1], neg_score: Bool = False, n_jobs: Int = 0, cv: Int = 5) -> Tuple[Dict[String, String], SIMD[float32, 1]]
```

Exhaustive search over specified parameter values for an estimator.

**Parameters:**

- **m_type** (`CVP`): Model type.

**Args:**

- **X** (`Matrix`): Samples.
- **y** (`PythonObject`): Targets.
- **param_grid** (`Dict`): Dictionary with parameters names as keys and lists of parameter settings to try as values.
- **scoring** (`fn(PythonObject, List[String]) raises -> SIMD[float32, 1]`): Scoring function.
- **neg_score** (`Bool`): Invert the scoring results when finding the best params. `-1` means using all processors.
- **n_jobs** (`Int`): Number of jobs to run in parallel.
- **cv** (`Int`): Number of folds in a KFold.

**Returns:**

`Tuple`: Best parameters.

**Raises:**

