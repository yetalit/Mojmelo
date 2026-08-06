Mojo function

# `GridSearchCV`

```mojo
fn def GridSearchCV[m_type: CV](X: Matrix, y: Matrix, param_grid: Dict[String, List[String]], scoring: def(Matrix, Matrix) raises thin -> Float32, neg_score: Bool = False, n_jobs: Int = Int(0), cv: Int = Int(5)) -> Tuple[Dict[String, String], Float32]
```

Exhaustive search over specified parameter values for an estimator.

**Parameters:**

- **m_type** (`CV`): Model type.

**Args:**

- **X** (`Matrix`): Samples.
- **y** (`Matrix`): Targets.
- **param_grid** (`Dict[String, List[String]]`): Dictionary with parameters names as keys and lists of parameter settings to try as values.
- **scoring** (`def(Matrix, Matrix) raises thin -> Float32`): Scoring function.
- **neg_score** (`Bool`): Invert the scoring results when finding the best params.
- **n_jobs** (`Int`): Number of jobs to run in parallel. `-1` means using all processors.
- **cv** (`Int`): Number of folds in a KFold.

**Returns:**

`Tuple[Dict[String, String], Float32]`: Best parameters.

**Raises:**

