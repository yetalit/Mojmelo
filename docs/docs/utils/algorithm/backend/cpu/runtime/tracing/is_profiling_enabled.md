Mojo function

# `is_profiling_enabled`

```mojo
fn def is_profiling_enabled[type: TraceCategory, level: TraceLevel]() -> Bool
```

Returns True if the profiling is enabled for that specific type and level and False otherwise.

**Parameters:**

- **type** (`TraceCategory`): The trace category to check.
- **level** (`TraceLevel`): The trace level to check.

**Returns:**

`Bool`: True if profiling is enabled for the specified type and level.

