Mojo function

# `MinMaxScaler`

```mojo
fn MinMaxScaler(data: Matrix, feature_range: Tuple[Float32, Float32] = Tuple(Float32("0"), Float32("1"))) -> Tuple[Matrix, Matrix, Matrix]
```

Transform features by scaling each feature to a given range.

**Args:**

- **data** (`Matrix`): Data.
- **feature_range** (`Tuple[Float32, Float32]`): Desired range of transformed data.

**Returns:**

`Tuple[Matrix, Matrix, Matrix]`: Scaled data, data_min, data_max.

**Raises:**

```mojo
fn MinMaxScaler(data: Matrix, x_min: Matrix, x_max: Matrix, feature_range: Tuple[Float32, Float32] = Tuple(Float32("0"), Float32("1"))) -> Matrix
```

Transform features by scaling each feature to a given range, data_min and data_max.

**Args:**

- **data** (`Matrix`): Data.
- **x_min** (`Matrix`): Per feature minimum seen in the data.
- **x_max** (`Matrix`): Per feature maximum seen in the data.
- **feature_range** (`Tuple[Float32, Float32]`): Desired range of transformed data.

**Returns:**

`Matrix`: Scaled data.

**Raises:**

