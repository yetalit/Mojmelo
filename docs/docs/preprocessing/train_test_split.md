Mojo function

# `train_test_split`

```mojo
fn train_test_split(X: Matrix, y: Matrix, *, test_size: SIMD[float16, 1] = 0.5, train_size: SIMD[float16, 1] = 0) -> Tuple[Matrix, Matrix, Matrix, Matrix]
```

Split matrices into random train and test subsets.

**Args:**

- **X** (`Matrix`)
- **y** (`Matrix`)
- **test_size** (`SIMD`)
- **train_size** (`SIMD`)

**Returns:**

`Tuple`

**Raises:**

```mojo
fn train_test_split(X: Matrix, y: Matrix, *, random_state: Int, test_size: SIMD[float16, 1] = 0.5, train_size: SIMD[float16, 1] = 0) -> Tuple[Matrix, Matrix, Matrix, Matrix]
```

Split matrices into random train and test subsets.

**Args:**

- **X** (`Matrix`)
- **y** (`Matrix`)
- **random_state** (`Int`)
- **test_size** (`SIMD`)
- **train_size** (`SIMD`)

**Returns:**

`Tuple`

**Raises:**

```mojo
fn train_test_split(X: Matrix, y: PythonObject, *, test_size: SIMD[float16, 1] = 0.5, train_size: SIMD[float16, 1] = 0) -> Tuple[Matrix, Matrix, SplittedPO]
```

Split matrix and python object into random train and test subsets.

**Args:**

- **X** (`Matrix`)
- **y** (`PythonObject`)
- **test_size** (`SIMD`)
- **train_size** (`SIMD`)

**Returns:**

`Tuple`

**Raises:**

```mojo
fn train_test_split(X: Matrix, y: PythonObject, *, random_state: Int, test_size: SIMD[float16, 1] = 0.5, train_size: SIMD[float16, 1] = 0) -> Tuple[Matrix, Matrix, SplittedPO]
```

Split matrix and python object into random train and test subsets.

**Args:**

- **X** (`Matrix`)
- **y** (`PythonObject`)
- **random_state** (`Int`)
- **test_size** (`SIMD`)
- **train_size** (`SIMD`)

**Returns:**

`Tuple`

**Raises:**

