Mojo struct

# `PCA`

```mojo
@memory_only
struct PCA
```

Principal component analysis (PCA). Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

## Aliases

- `__del__is_trivial = False`

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

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, n_components: Int, whiten: Bool = False, lapack: Bool = False)
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
fn fit(mut self, X: Matrix)
```

Fit the model.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Raises:**

### `transform`

```mojo
fn transform(self, X: Matrix) -> Matrix
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
fn inverse_transform(self, X_transformed: Matrix) -> Matrix
```

Transform data back to its original space.

**Args:**

- **self** (`Self`)
- **X_transformed** (`Matrix`)

**Returns:**

`Matrix`: Original data.

**Raises:**


