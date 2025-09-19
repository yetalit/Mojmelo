Mojo struct

# `Node`

```mojo
@memory_only
struct Node
```

## Fields

- **feature** (`Int`)
- **threshold** (`SIMD[float32, 1]`)
- **left** (`UnsafePointer[Node]`)
- **right** (`UnsafePointer[Node]`)
- **value** (`SIMD[float32, 1]`)

## Implemented traits

`AnyType`, `Copyable`, `Movable`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, feature: Int = -1, threshold: SIMD[float32, 1] = 0, left: UnsafePointer[Node] = UnsafePointer[Node](0), right: UnsafePointer[Node] = UnsafePointer[Node](0), value: SIMD[float32, 1] = inf[::DType]())
```

**Args:**

- **feature** (`Int`)
- **threshold** (`SIMD`)
- **left** (`UnsafePointer`)
- **right** (`UnsafePointer`)
- **value** (`SIMD`)
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


