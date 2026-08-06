Mojo function

# `parallelize`

```mojo
fn def parallelize[origins: OriginSet, //, func: def(Int) capturing thin -> None](num_work_items: Int)
```

Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and returns when all are complete.

**Parameters:**

- **origins** (`OriginSet`): The capture origins.
- **func** (`def(Int) capturing thin -> None`): The function to invoke.

**Args:**

- **num_work_items** (`Int`): Number of parallel tasks.

```mojo
fn def parallelize[origins: OriginSet, //, func: def(Int) capturing thin -> None](num_work_items: Int, num_workers: Int)
```

Executes func(0) ... func(num_work_items-1) as sub-tasks in parallel, and returns when all are complete.

**Parameters:**

- **origins** (`OriginSet`): The capture origins.
- **func** (`def(Int) capturing thin -> None`): The function to invoke.

**Args:**

- **num_work_items** (`Int`): Number of parallel tasks.
- **num_workers** (`Int`): The number of workers to use for execution.

