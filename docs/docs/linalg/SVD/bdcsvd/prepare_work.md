Mojo function

# `prepare_work`

```mojo
fn def prepare_work(A: Mat, do_transpose: Bool) -> Mat
```

Returns a fresh Mat in both branches (transpose or plain copy) — avoids assigning Mat by copy.

**Args:**

- **A** (`Mat`)
- **do_transpose** (`Bool`)

**Returns:**

`Mat`

