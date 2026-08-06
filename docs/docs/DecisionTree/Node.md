Mojo struct

# `Node`

```mojo
@memory_only
struct Node
```

## Fields

- **feature** (`Int`)
- **threshold** (`Float32`)
- **left** (`Optional[Pointer[Node, MutUntrackedOrigin]]`)
- **right** (`Optional[Pointer[Node, MutUntrackedOrigin]]`)
- **value** (`Float32`)

## Implemented traits

`AnyType`, `Copyable`, `Deinitable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, feature: Int = Int(-1), threshold: Float32 = 0, left: Optional[Pointer[Self, MutUntrackedOrigin]] = None, right: Optional[Pointer[Self, MutUntrackedOrigin]] = None, value: Float32 = inf[DType.float32]())
```

**Args:**

- **feature** (`Int`)
- **threshold** (`Float32`)
- **left** (`Optional[Pointer[Self, MutUntrackedOrigin]]`)
- **right** (`Optional[Pointer[Self, MutUntrackedOrigin]]`)
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


