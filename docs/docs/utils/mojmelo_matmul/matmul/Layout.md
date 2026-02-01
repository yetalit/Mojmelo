Mojo struct

# `Layout`

```mojo
@register_passable_trivial
struct Layout
```

## Aliases

- `__del__is_trivial = True`
- `__moveinit__is_trivial = True`
- `__copyinit__is_trivial = True`

## Fields

- **shape** (`IndexList[2]`)
- **strides** (`IndexList[2]`)

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Movable`, `Writable`

## Methods

### `__init__`

```mojo
fn __init__(shape: Tuple[Int, Int], strides: Tuple[Int, Int]) -> Self
```

**Args:**

- **shape** (`Tuple`)
- **strides** (`Tuple`)

**Returns:**

`Self`

```mojo
fn __init__(shape: Tuple[Int, Int]) -> Self
```

**Args:**

- **shape** (`Tuple`)

**Returns:**

`Self`

### `__call__`

```mojo
fn __call__(self, i: Int, j: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

**Returns:**

`Int`

### `size`

```mojo
fn size(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `write_to`

```mojo
fn write_to[W: Writer](self, mut writer: W)
```

**Parameters:**

- **W** (`Writer`)

**Args:**

- **self** (`Self`)
- **writer** (`W`)


