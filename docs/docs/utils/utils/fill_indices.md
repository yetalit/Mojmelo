Mojo function

# `fill_indices`

```mojo
fn def fill_indices(N: Int, ptr: Optional[Pointer[Int, MutUntrackedOrigin]] = None) -> Pointer[Int, MutUntrackedOrigin]
```

Generates indices from 0 to N.

**Args:**

- **N** (`Int`)
- **ptr** (`Optional[Pointer[Int, MutUntrackedOrigin]]`)

**Returns:**

`Pointer[Int, MutUntrackedOrigin]`: The pointer to indices.

**Raises:**

