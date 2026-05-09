Mojo struct

# `DBSCAN`

```mojo
@memory_only
struct DBSCAN[metric: String = "euc"]
```

A density based clustering method that expands clusters from samples that have more neighbors within a radius.

## Parameters

- **metric** (`String`): Metric to use for distance computation:
    Euclidean -> 'euc';
    Manhattan -> 'man'.

## Fields

- **eps** (`Float32`): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- **min_samples** (`Int`): The number of samples in a neighborhood for a point to be considered as a core point.
- **labels** (`List[Int]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, eps: Float32 = 1, min_samples: Int = 5)
```

**Args:**

- **eps** (`Float32`)
- **min_samples** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `fit`

```mojo
fn fit(mut self, X: Matrix)
```

Perform clustering.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Raises:**

### `fit_predict`

```mojo
fn fit_predict(mut self, X: Matrix) -> List[Int]
```

Perform clustering and predict cluster indices.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`List[Int]`: List of cluster indices.

**Raises:**


