Mojo struct

# `Node`

```mojo
@memory_only
struct Node
```

## Fields

- **feature** (`Int`)
- **threshold** (`Float32`)
- **left** (`UnsafePointer[Node, MutAnyOrigin]`)
- **right** (`UnsafePointer[Node, MutAnyOrigin]`)
- **value** (`Float32`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
def __init__(out self, feature: Int = -1, threshold: Float32 = 0, left: UnsafePointer[Node, MutAnyOrigin] = UnsafePointer(), right: UnsafePointer[Node, MutAnyOrigin] = UnsafePointer(), value: Float32 = inf[DType.float32]())
```

**Args:**

- **feature** (`Int`)
- **threshold** (`Float32`)
- **left** (`UnsafePointer`)
- **right** (`UnsafePointer`)
- **value** (`Float32`)
- **self** (`Self`)

**Returns:**

`Self`

### `is_leaf_node`

```mojo
def is_leaf_node(self) -> Bool
```

**Args:**

- **self** (`Self`)

**Returns:**

`Bool`

### `__str__`

```mojo
def __str__(self) -> String
```

**Args:**

- **self** (`Self`)

**Returns:**

`String`


