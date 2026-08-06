Mojo struct

# `Trace`

```mojo
@memory_only
struct Trace[level: TraceLevel, *, category: TraceCategory = TraceCategory.MAX, target: Optional[StringSpan[ImmStaticOrigin]] = None]
```

An object representing a specific trace.

This struct provides functionality for creating and managing trace events
for profiling and debugging purposes.

## Parameters

- **level** (`TraceLevel`): The trace level to use.
- **category** (`TraceCategory`): The trace category to use (defaults to TraceCategory.MAX).
- **target** (`Optional[StringSpan[ImmStaticOrigin]]`): Optional target information to include in the trace.

## Fields

- **int_payload** (`OptionalReg[Int]`): Optional integer payload, typically used for task IDs that are appended to trace names.
- **detail** (`String`): Additional details about the trace event, included when detailed tracing is enabled.
- **event_id** (`Int`): Unique identifier for the trace event, assigned when the trace begins.
- **parent_id** (`Int`): Identifier of the parent trace event, used for creating hierarchical trace relationships.
- **color** (`Optional[Color]`): Color of the trace span in NSight Systems viewer, only used for NVTX markers.

## Implemented traits

`AnyType`, `Copyable`, `Deinitable`, `ImplicitlyCopyable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, *, var _name_value: Variant[String, StringSpan[ImmStaticOrigin]], detail: String = "", parent_id: Int = Int(0), task_id: OptionalReg[Int] = None, color: Optional[Color] = None)
```

Creates a Mojo trace with the given name.

**Args:**

- **_name_value** (`Variant[String, StringSpan[ImmStaticOrigin]]`): The name that is used to identify this Mojo trace.
- **detail** (`String`): Details of the trace entry.
- **parent_id** (`Int`): Parent to associate the trace with. Trace name will be
    appended to parent name. 0 (default) indicates no parent.
- **task_id** (`OptionalReg[Int]`): Int that is appended to name.
- **color** (`Optional[Color]`): Color of the trace span when visualized.
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, var name: String, detail: String = "", parent_id: Int = Int(0), color: Optional[Color] = None, *, task_id: OptionalReg[Int] = None)
```

Creates a Mojo trace with the given string name.

**Args:**

- **name** (`String`): The name that is used to identify this Mojo trace.
- **detail** (`String`): Details of the trace entry.
- **parent_id** (`Int`): Parent to associate the trace with. Trace name will be
    appended to parent name. 0 (default) indicates no parent.
- **color** (`Optional[Color]`): Color of the trace span when visualized.
- **task_id** (`OptionalReg[Int]`): Int that is appended to name.
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, name: StringSpan[ImmStaticOrigin], detail: String = "", parent_id: Int = Int(0), color: Optional[Color] = None, *, task_id: OptionalReg[Int] = None)
```

Creates a Mojo trace with the given static string name.

**Args:**

- **name** (`StringSpan[ImmStaticOrigin]`): The name that is used to identify this Mojo trace.
- **detail** (`String`): Details of the trace entry.
- **parent_id** (`Int`): Parent to associate the trace with. Trace name will be
    appended to parent name. 0 (default) indicates no parent.
- **color** (`Optional[Color]`): Color of the trace span when visualized.
- **task_id** (`OptionalReg[Int]`): Int that is appended to name.
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, name: StringLiteral, detail: String = "", parent_id: Int = Int(0), color: Optional[Color] = None, *, task_id: OptionalReg[Int] = None)
```

Creates a Mojo trace with the given string literal name.

**Args:**

- **name** (`StringLiteral`): The name that is used to identify this Mojo trace.
- **detail** (`String`): Details of the trace entry.
- **parent_id** (`Int`): Parent to associate the trace with. Trace name will be
    appended to parent name. 0 (default) indicates no parent.
- **color** (`Optional[Color]`): Color of the trace span when visualized.
- **task_id** (`OptionalReg[Int]`): Int that is appended to name.
- **self** (`Self`)

**Returns:**

`Self`

### `__enter__`

```mojo
fn def __enter__(mut self)
```

Enters the trace context.

This begins recording of the trace event.

**Args:**

- **self** (`Self`)

**Raises:**

If the operation fails.

### `__exit__`

```mojo
fn def __exit__(self)
```

Exits the trace context.

This finishes recording of the trace event.

**Args:**

- **self** (`Self`)

### `name`

```mojo
fn def name(self) -> String
```

Returns the name of the trace.

**Args:**

- **self** (`Self`)

**Returns:**

`String`: The name of the trace as a String.

### `start`

```mojo
fn def start(mut self)
```

Start recording trace event.

This begins recording of the trace event, similar to __enter__.

**Args:**

- **self** (`Self`)

**Raises:**

If the operation fails.

### `end`

```mojo
fn def end(mut self)
```

End recording trace event.

This finishes recording of the trace event, similar to __exit__.

**Args:**

- **self** (`Self`)

**Raises:**

If the operation fails.


