Mojo struct

# `TraceLevel`

```mojo
@register_passable_trivial
struct TraceLevel
```

An enum-like struct specifying the level of tracing to perform.

## Aliases

- `ALWAYS = TraceLevel(Int(0))`: Always trace at this level.
- `OP = TraceLevel(Int(1))`: Operation-level tracing.
- `THREAD = TraceLevel(Int(2))`: Thread-level tracing.

## Fields

- **value** (`Int`): The integer value representing the trace level. Lower values indicate higher priority trace levels:
- 0 (ALWAYS): Always traced
- 1 (OP): Operation-level tracing
- 2 (THREAD): Thread-level tracing

## Implemented traits

`AnyType`, `Comparable`, `Copyable`, `Deinitable`, `Equatable`, `ImplicitlyCopyable`, `Movable`, `RegisterPassable`, `TrivialRegisterPassable`

## Methods

### `__init__`

```mojo
fn def __init__(value: Int) -> Self
```

Initializes a TraceLevel with the given integer value.

**Args:**

- **value** (`Int`): The integer value for the trace level.

**Returns:**

`Self`

### `__lt__`

```mojo
fn def __lt__(self, rhs: Self) -> Bool
```

Performs less than comparison.

**Args:**

- **self** (`Self`)
- **rhs** (`Self`): The value to compare.

**Returns:**

`Bool`: True if this value is less than to `rhs`.

### `__eq__`

```mojo
fn def __eq__(self, rhs: Self) -> Bool
```

Compares for equality.

**Args:**

- **self** (`Self`)
- **rhs** (`Self`): The value to compare.

**Returns:**

`Bool`: True if they are equal.

### `__int__`

```mojo
fn def __int__(self) -> Int
```

Converts the trace level to an integer.

**Args:**

- **self** (`Self`)

**Returns:**

`Int`: The integer value of the trace level.


