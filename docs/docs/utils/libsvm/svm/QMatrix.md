Mojo trait

# `QMatrix`

## Implemented traits

`AnyType`

## Methods

### `get_Q`

```mojo
fn def get_Q(mut self: _Self, column: Int, _len: Int) -> UnsafePointer[Float32, MutUntrackedOrigin]
```

**Args:**

- **self** (`_Self`)
- **column** (`Int`)
- **_len** (`Int`)

**Returns:**

`UnsafePointer[Float32, MutUntrackedOrigin]`

### `get_QD`

```mojo
fn def get_QD(self: _Self) -> UnsafePointer[Float64, MutUntrackedOrigin]
```

**Args:**

- **self** (`_Self`)

**Returns:**

`UnsafePointer[Float64, MutUntrackedOrigin]`

### `swap_index`

```mojo
fn def swap_index(mut self: _Self, i: Int, j: Int)
```

**Args:**

- **self** (`_Self`)
- **i** (`Int`)
- **j** (`Int`)


