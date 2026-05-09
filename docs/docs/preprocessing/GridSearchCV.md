Mojo function

# `GridSearchCV`

```mojo
fn GridSearchCV[m_type: CV, scoring: String](X: Matrix, y: Matrix, param_grid: Dict[String, List[String]], neg_score: Bool = False, n_jobs: Int = 0, cv: Int = 5) -> Tuple[Dict[String, String], Float32]
```

Exhaustive search over specified parameter values for an estimator.

**Parameters:**

- **m_type** (`CV`): Model type.
- **scoring** (`String`): The scoring function:
    accuracy_score -> 'accuracy';
    r2_score -> 'r2';
    MSE -> 'mse'.

**Args:**

- **X** (`Matrix`): Samples.
- **y** (`Matrix`): Targets.
- **param_grid** (`Dict[String, List[String]]`): Dictionary with parameters names as keys and lists of parameter settings to try as values.
- **neg_score** (`Bool`): Invert the scoring results when finding the best params.
- **n_jobs** (`Int`): Number of jobs to run in parallel. `-1` means using all processors.
- **cv** (`Int`): Number of folds in a KFold.

**Returns:**

`Tuple[Dict[String, String], Float32]`: Best parameters.

**Raises:**

