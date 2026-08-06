Mojo struct

# `Color`

```mojo
@register_passable_trivial
struct Color
```

## Aliases

- `FORMAT = 1`
- `MODULAR_PURPLE = Color(Int(11909877))`
- `BLUE = Color(Int(255))`
- `GREEN = Color(Int(32768))`
- `ORANGE = Color(Int(16753920))`
- `PURPLE = Color(Int(8388736))`
- `RED = Color(Int(16711680))`
- `WHITE = Color(Int(16777215))`
- `YELLOW = Color(Int(16776960))`

## Implemented traits

`AnyType`, `Copyable`, `Deinitable`, `ImplicitlyCopyable`, `Intable`, `Movable`, `RegisterPassable`, `TrivialRegisterPassable`

## Methods

### `__init__`

```mojo
fn def __init__(colorname: StringSpan[ImmStaticOrigin]) -> Self
```

Initialize Color from a StaticString color name.

**Args:**

- **colorname** (`StringSpan[ImmStaticOrigin]`): The name of the color to use.

**Returns:**

`Self`

### `__int__`

```mojo
fn def __int__(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`


