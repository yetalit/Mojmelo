Mojo trait

# `QMatrix`

## Implemented traits

`AnyType`

## Methods

### `get_Q`

```mojo
fn def get_Q(mut self, column: Int, _len: Int) -> Pointer[Float32, MutUntrackedOrigin]
```

**Args:**

- **self** (`_Self`)
- **column** (`Int`)
- **_len** (`Int`)

**Returns:**

`Pointer[Float32, MutUntrackedOrigin]`

### `get_QD`

```mojo
fn def get_QD(self) -> Pointer[Float64, MutUntrackedOrigin]
```

**Args:**

- **self** (`_Self`)

**Returns:**

`Pointer[Float64, MutUntrackedOrigin]`

### `swap_index`

```mojo
fn def swap_index(mut self, i: Int, j: Int)
```

**Args:**

- **self** (`_Self`)
- **i** (`Int`)
- **j** (`Int`)


