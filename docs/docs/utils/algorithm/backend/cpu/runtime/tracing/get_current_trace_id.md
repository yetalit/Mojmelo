Mojo function

# `get_current_trace_id`

```mojo
fn def get_current_trace_id[level: TraceLevel]() -> Int
```

Returns the id of last created trace entry on the current thread.

**Parameters:**

- **level** (`TraceLevel`): The trace level to check.

**Returns:**

`Int`: The ID of the current trace if profiling is enabled, otherwise 0.

