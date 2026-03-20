Mojo struct

# `PCA`

```mojo
@memory_only
struct PCA
```

Principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

## Aliases

- `MODEL_ID = 12`

## Fields

- **n_components** (`Int`): Number of components to keep.
- **components** (`Matrix`)
- **components_T** (`Matrix`)
- **explained_variance** (`Matrix`): The amount of variance explained by each of the selected components.
- **explained_variance_ratio** (`Matrix`): Percentage of variance explained by each of the selected components.
- **mean** (`Matrix`)
- **whiten** (`Bool`): To transform data to have zero mean, unit variance, and no correlation between features.
- **whiten_** (`Matrix`)
- **lapack** (`Bool`): Use LAPACK to calculate svd.

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
def __init__(out self, n_components: Int, whiten: Bool = False, lapack: Bool = False)
```

**Args:**

- **n_components** (`Int`)
- **whiten** (`Bool`)
- **lapack** (`Bool`)
- **self** (`Self`)

**Returns:**

`Self`

### `fit`

```mojo
def fit(mut self, X: Matrix)
```

Fit the model.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Raises:**

### `transform`

```mojo
def transform(self, X: Matrix) -> Matrix
```

Apply dimensionality reduction to X. X is projected on the first principal components previously extracted from a training set.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: Projection of X in the first principal components.

**Raises:**

### `inverse_transform`

```mojo
def inverse_transform(self, X_transformed: Matrix) -> Matrix
```

Transform data back to its original space.

**Args:**

- **self** (`Self`)
- **X_transformed** (`Matrix`)

**Returns:**

`Matrix`: Original data.

**Raises:**

### `save`

```mojo
def save(self, path: String)
```

Save model data necessary for transformation to the specified path.

**Args:**

- **self** (`Self`)
- **path** (`String`)

**Raises:**

### `load`

```mojo
@staticmethod
def load(path: String) -> Self
```

Load a saved model from the specified path for transformation.

**Args:**

- **path** (`String`)

**Returns:**

`Self`

**Raises:**


