Mojo trait

# `QMatrix`

## Implemented traits

`AnyType`

## Methods

### `get_Q`

```mojo
fn get_Q(mut self: _Self, column: Int, _len: Int) -> Optional[UnsafePointer[Float32, MutExternalOrigin]]
```

**Args:**

- **self** (`_Self`)
- **column** (`Int`)
- **_len** (`Int`)

**Returns:**

`Optional[UnsafePointer[Float32, MutExternalOrigin]]`

### `get_QD`

```mojo
fn get_QD(self: _Self) -> Optional[UnsafePointer[Float64, MutExternalOrigin]]
```

**Args:**

- **self** (`_Self`)

**Returns:**

`Optional[UnsafePointer[Float64, MutExternalOrigin]]`

### `swap_index`

```mojo
fn swap_index(mut self: _Self, i: Int, j: Int)
```

**Args:**

- **self** (`_Self`)
- **i** (`Int`)
- **j** (`Int`)


