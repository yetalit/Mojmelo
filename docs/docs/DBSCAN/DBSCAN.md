Mojo struct

# `DBSCAN`

```mojo
@memory_only
struct DBSCAN
```

A density based clustering method that expands clusters from samples that have more neighbors within a radius.

## Fields

- **squared_eps** (`SIMD[float32, 1]`): The maximum squared distance between two samples for one to be considered as in the neighborhood of the other.
- **min_samples** (`Int`): The number of samples in a neighborhood for a point to be considered as a core point.

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, eps: SIMD[float32, 1] = 1, min_samples: Int = 5)
```

**Args:**

- **eps** (`SIMD`)
- **min_samples** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `predict`

```mojo
fn predict(mut self, X: Matrix) -> Matrix
```

Predict cluster indices.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: Vector of cluster indices.

**Raises:**


