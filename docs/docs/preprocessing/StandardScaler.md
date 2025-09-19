Mojo function

# `StandardScaler`

```mojo
fn StandardScaler(data: Matrix) -> Tuple[Matrix, Matrix, Matrix]
```

Standardize features by removing the mean and scaling to unit variance.

**Args:**

- **data** (`Matrix`): Data.

**Returns:**

`Tuple`: Scaled data, mean, standard deviation.

**Raises:**

```mojo
fn StandardScaler(data: Matrix, mu: Matrix, sigma: Matrix) -> Matrix
```

Standardize features by removing the mean and scaling to unit variance given mean and standard deviation.

**Args:**

- **data** (`Matrix`): Data.
- **mu** (`Matrix`): Mean.
- **sigma** (`Matrix`): Standard Deviation.

**Returns:**

`Matrix`: Scaled data.

**Raises:**

