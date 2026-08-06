Mojo function

# `sync_parallelize`

```mojo
fn def sync_parallelize[origins: OriginSet, //, func: def(Int) raises capturing thin -> None](num_work_items: Int)
```

Executes func(0) ... func(num_work_items-1) as parallel sub-tasks, and returns when all are complete.

TODO: Currently exceptions raised by func will cause a trap rather than
      be propagated back to the caller.

**Parameters:**

- **origins** (`OriginSet`): The capture origins.
- **func** (`def(Int) raises capturing thin -> None`): The function to invoke.

**Args:**

- **num_work_items** (`Int`): Number of parallel tasks.

