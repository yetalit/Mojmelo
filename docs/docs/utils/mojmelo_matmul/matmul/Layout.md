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

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDeletable`, `Movable`, `RegisterPassable`, `TrivialRegisterPassable`, `Writable`

## Methods

### `__init__`

```mojo
fn def __init__(shape: Tuple[Int, Int], strides: Tuple[Int, Int]) -> Self
```

**Args:**

- **shape** (`Tuple[Int, Int]`)
- **strides** (`Tuple[Int, Int]`)

**Returns:**

`Self`

```mojo
fn def __init__(shape: Tuple[Int, Int]) -> Self
```

**Args:**

- **shape** (`Tuple[Int, Int]`)

**Returns:**

`Self`

### `__call__`

```mojo
fn def __call__(self, i: Int, j: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

**Returns:**

`Int`

### `size`

```mojo
fn def size(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `write_to`

```mojo
fn def write_to[W: Writer](self, mut writer: W)
```

**Parameters:**

- **W** (`Writer`)

**Args:**

- **self** (`Self`)
- **writer** (`W`)


