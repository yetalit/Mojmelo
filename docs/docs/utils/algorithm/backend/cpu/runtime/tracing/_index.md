Mojo module

# `tracing`

Provides tracing utilities.

## Aliases

- `log = Logger(stderr, prefix=String("[OP] "), source_location=False)`: Logger instance for operation tracing with INFO level and [OP] prefix.

## Structs

- [`Color`](Color.md)
- [`TraceCategory`](TraceCategory.md): An enum-like struct specifying the type of tracing to perform.
- [`TraceLevel`](TraceLevel.md): An enum-like struct specifying the level of tracing to perform.
- [`Trace`](Trace.md): An object representing a specific trace.

## Functions

- [`is_profiling_enabled`](is_profiling_enabled.md)
- [`is_profiling_disabled`](is_profiling_disabled.md)
- [`trace_arg`](trace_arg.md)
- [`get_current_trace_id`](get_current_trace_id.md)

