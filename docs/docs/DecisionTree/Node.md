Mojo struct

# `Node`

```mojo
@memory_only
struct Node
```

## Aliases

- `__del__is_trivial = True`
- `__moveinit__is_trivial = True`
- `__copyinit__is_trivial = True`

## Fields

- **feature** (`Int`)
- **threshold** (`Float32`)
- **left** (`UnsafePointer[Node]`)
- **right** (`UnsafePointer[Node]`)
- **value** (`Float32`)

## Implemented traits

`AnyType`, `Copyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, feature: Int = -1, threshold: Float32 = 0, left: UnsafePointer[Node] = UnsafePointer[Node, AddressSpace(0), True, MutableAnyOrigin](), right: UnsafePointer[Node] = UnsafePointer[Node, AddressSpace(0), True, MutableAnyOrigin](), value: Float32 = inf[DType.float32]())
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
fn is_leaf_node(self) -> Bool
```

**Args:**

- **self** (`Self`)

**Returns:**

`Bool`

### `__str__`

```mojo
fn __str__(self) -> String
```

**Args:**

- **self** (`Self`)

**Returns:**

`String`


