Mojo function

# `train_test_split`

```mojo
fn train_test_split(X: Matrix, y: Matrix, *, test_size: Float64 = 0.5, train_size: Float64 = 0) -> Tuple[Matrix, Matrix, Matrix, Matrix]
```

Split matrices into random train and test subsets.

**Args:**

- **X** (`Matrix`)
- **y** (`Matrix`)
- **test_size** (`Float64`)
- **train_size** (`Float64`)

**Returns:**

`Tuple`

**Raises:**

```mojo
fn train_test_split(X: Matrix, y: Matrix, *, random_state: Int, test_size: Float64 = 0.5, train_size: Float64 = 0) -> Tuple[Matrix, Matrix, Matrix, Matrix]
```

Split matrices into random train and test subsets.

**Args:**

- **X** (`Matrix`)
- **y** (`Matrix`)
- **random_state** (`Int`)
- **test_size** (`Float64`)
- **train_size** (`Float64`)

**Returns:**

`Tuple`

**Raises:**

