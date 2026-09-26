Mojo struct

# `BDecisionTree`

```mojo
@memory_only
struct BDecisionTree
```

## Fields

- **min_samples_split** (`Int`)
- **max_depth** (`Int`)
- **reg_lambda** (`Float32`)
- **reg_alpha** (`Float32`)
- **gamma** (`Float32`)
- **n_bins** (`Int`)
- **root** (`Optional[Pointer[Node, MutUntrackedOrigin]]`)

## Implemented traits

`AnyType`, `Copyable`, `Deinitable`, `ImplicitlyCopyable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, min_samples_split: Int = Int(10), max_depth: Int = Int(3), reg_lambda: Float32 = 1, reg_alpha: Float32 = 0, gamma: Float32 = 0, n_bins: Int = Int(0))
```

**Args:**

- **min_samples_split** (`Int`)
- **max_depth** (`Int`)
- **reg_lambda** (`Float32`)
- **reg_alpha** (`Float32`)
- **gamma** (`Float32`)
- **n_bins** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `__deinit__`

```mojo
fn def __deinit__(deinit self)
```

**Args:**

- **self** (`Self`)

### `fit`

```mojo
fn def fit(mut self, X: Matrix, g: Matrix, h: Matrix)
```

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **g** (`Matrix`)
- **h** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn def predict(self, X: Matrix) -> Matrix
```

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`

**Raises:**


