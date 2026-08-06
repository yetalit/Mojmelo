Mojo function

# `is_profiling_disabled`

```mojo
fn def is_profiling_disabled[type: TraceCategory, level: TraceLevel]() -> Bool
```

Returns False if the profiling is enabled for that specific type and level and True otherwise.

**Parameters:**

- **type** (`TraceCategory`): The trace category to check.
- **level** (`TraceLevel`): The trace level to check.

**Returns:**

`Bool`: True if profiling is disabled for the specified type and level.

