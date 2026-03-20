Mojo struct

# `TreeUnionFind`

```mojo
@memory_only
struct TreeUnionFind
```

## Aliases

- `width = 2`

## Fields

- **size** (`Int`)
- **is_component** (`List[Bool]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
def __init__(out self, size: Int)
```

**Args:**

- **size** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `union_`

```mojo
def union_(mut self, x: Scalar[DType.int], y: Scalar[DType.int])
```

**Args:**

- **self** (`Self`)
- **x** (`Scalar`)
- **y** (`Scalar`)

### `find`

```mojo
def find(mut self, x: Scalar[DType.int]) -> Scalar[DType.int]
```

**Args:**

- **self** (`Self`)
- **x** (`Scalar`)

**Returns:**

`Scalar`

### `components`

```mojo
def components(self) -> List[Int]
```

**Args:**

- **self** (`Self`)

**Returns:**

`List`


