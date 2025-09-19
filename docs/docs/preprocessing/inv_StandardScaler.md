Mojo function

# `inv_StandardScaler`

```mojo
fn inv_StandardScaler(z: Matrix, mu: Matrix, sigma: Matrix) -> Matrix
```

Reproduce scaled data given its mean and standard deviation.

**Args:**

- **z** (`Matrix`): Scaled data.
- **mu** (`Matrix`): Mean.
- **sigma** (`Matrix`): Standard Deviation.

**Returns:**

`Matrix`: Original data.

**Raises:**

