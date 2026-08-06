Mojo struct

# `TraceCategory`

```mojo
@register_passable_trivial
struct TraceCategory
```

An enum-like struct specifying the type of tracing to perform.

## Aliases

- `OTHER = TraceCategory(Int(0))`: Other or uncategorized trace events.
- `ASYNCRT = TraceCategory(Int(1))`: Asynchronous runtime trace events.
- `MEM = TraceCategory(Int(2))`: Memory-related trace events.
- `Kernel = TraceCategory(Int(3))`: Kernel execution trace events.
- `MAX = TraceCategory(Int(4))`: MAX framework trace events.

## Fields

- **value** (`Int`): The integer value representing the trace category. Used for bitwise operations when determining if profiling is enabled for a specific category.

## Implemented traits

`AnyType`, `Copyable`, `Deinitable`, `Equatable`, `ImplicitlyCopyable`, `Intable`, `Movable`, `RegisterPassable`, `TrivialRegisterPassable`

## Methods

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

### `__ne__`

```mojo
fn def __ne__(self, rhs: Self) -> Bool
```

Compares for inequality.

**Args:**

- **self** (`Self`)
- **rhs** (`Self`): The value to compare.

**Returns:**

`Bool`: True if they are not equal.

### `__int__`

```mojo
fn def __int__(self) -> Int
```

Converts the trace category to an integer.

**Args:**

- **self** (`Self`)

**Returns:**

`Int`: The integer value of the trace category.


