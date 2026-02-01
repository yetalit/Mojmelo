Mojo struct

# `TreeUnionFind`

```mojo
@memory_only
struct TreeUnionFind
```

## Aliases

- `width = 2`
- `__del__is_trivial = False`

## Fields

- **size** (`Int`)
- **is_component** (`List[Bool]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, size: Int)
```

**Args:**

- **size** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `union_`

```mojo
fn union_(mut self, x: Int, y: Int)
```

**Args:**

- **self** (`Self`)
- **x** (`Int`)
- **y** (`Int`)

### `find`

```mojo
fn find(mut self, x: Scalar[DType.index]) -> Scalar[DType.index]
```

**Args:**

- **self** (`Self`)
- **x** (`Scalar`)

**Returns:**

`Scalar`

### `components`

```mojo
fn components(self) -> List[Int]
```

**Args:**

- **self** (`Self`)

**Returns:**

`List`


