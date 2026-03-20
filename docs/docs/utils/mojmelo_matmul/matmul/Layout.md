Mojo struct

# `Layout`

```mojo
@register_passable_trivial
struct Layout
```

## Fields

- **shape** (`IndexList[2]`)
- **strides** (`IndexList[2]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Movable`, `RegisterPassable`, `TrivialRegisterPassable`, `Writable`

## Methods

### `__init__`

```mojo
def __init__(shape: Tuple[Int, Int], strides: Tuple[Int, Int]) -> Self
```

**Args:**

- **shape** (`Tuple`)
- **strides** (`Tuple`)

**Returns:**

`Self`

```mojo
def __init__(shape: Tuple[Int, Int]) -> Self
```

**Args:**

- **shape** (`Tuple`)

**Returns:**

`Self`

### `__call__`

```mojo
def __call__(self, i: Int, j: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

**Returns:**

`Int`

### `size`

```mojo
def size(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `write_to`

```mojo
def write_to[W: Writer](self, mut writer: W)
```

**Parameters:**

- **W** (`Writer`)

**Args:**

- **self** (`Self`)
- **writer** (`W`)


