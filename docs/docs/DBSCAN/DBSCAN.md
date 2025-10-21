Mojo struct

# `DBSCAN`

```mojo
@memory_only
struct DBSCAN
```

A density based clustering method that expands clusters from samples that have more neighbors within a radius.

## Aliases

- `__del__is_trivial = False`

## Fields

- **eps** (`Float32`): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- **min_samples** (`Int`): The number of samples in a neighborhood for a point to be considered as a core point.
- **metric** (`String`): Metric to use for distance computation: Euclidean -> 'euc'; Manhattan -> 'man'.

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, eps: Float32 = 1, min_samples: Int = 5, metric: String = "euc")
```

**Args:**

- **eps** (`Float32`)
- **min_samples** (`Int`)
- **metric** (`String`)
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


