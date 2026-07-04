Mojo struct

# `Node`

```mojo
@memory_only
struct Node
```

## Fields

- **feature** (`Int`)
- **threshold** (`Float32`)
- **left** (`Optional[UnsafePointer[Node, MutAnyOrigin]]`)
- **right** (`Optional[UnsafePointer[Node, MutAnyOrigin]]`)
- **value** (`Float32`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyDeletable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, feature: Int = -1, threshold: Float32 = 0, left: Optional[UnsafePointer[Self, MutAnyOrigin]] = None, right: Optional[UnsafePointer[Self, MutAnyOrigin]] = None, value: Float32 = inf[DType.float32]())
```

**Args:**

- **feature** (`Int`)
- **threshold** (`Float32`)
- **left** (`Optional[UnsafePointer[Self, MutAnyOrigin]]`)
- **right** (`Optional[UnsafePointer[Self, MutAnyOrigin]]`)
- **value** (`Float32`)
- **self** (`Self`)

**Returns:**

`Self`

### `is_leaf_node`

```mojo
fn def is_leaf_node(self) -> Bool
```

**Args:**

- **self** (`Self`)

**Returns:**

`Bool`

### `__str__`

```mojo
fn def __str__(self) -> String
```

**Args:**

- **self** (`Self`)

**Returns:**

`String`


