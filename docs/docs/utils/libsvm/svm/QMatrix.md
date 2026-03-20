Mojo trait

# `QMatrix`

## Implemented traits

`AnyType`

## Methods

### `get_Q`

```mojo
def get_Q(mut self: _Self, column: Int, _len: Int) -> UnsafePointer[Float32, MutExternalOrigin]
```

**Args:**

- **self** (`_Self`)
- **column** (`Int`)
- **_len** (`Int`)

**Returns:**

`UnsafePointer`

### `get_QD`

```mojo
def get_QD(self: _Self) -> UnsafePointer[Float64, MutExternalOrigin]
```

**Args:**

- **self** (`_Self`)

**Returns:**

`UnsafePointer`

### `swap_index`

```mojo
def swap_index(mut self: _Self, i: Int, j: Int)
```

**Args:**

- **self** (`_Self`)
- **i** (`Int`)
- **j** (`Int`)


