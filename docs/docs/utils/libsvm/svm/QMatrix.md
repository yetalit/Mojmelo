Mojo trait

# `QMatrix`

## Aliases

- `__del__is_trivial = `: A flag (often compiler generated) to indicate whether the implementation of `__del__` is trivial. The implementation of `__del__` is considered to be trivial if:
- The struct has a compiler-generated trivial destructor and all its fields
  have a trivial `__del__` method.

In practice, it means that the `__del__` can be considered as no-op.

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `get_Q`

```mojo
fn get_Q(mut self: _Self, column: Int, _len: Int) -> UnsafePointer[Float32, origin_of(MutOrigin.external)]
```

**Args:**

- **self** (`_Self`)
- **column** (`Int`)
- **_len** (`Int`)

**Returns:**

`UnsafePointer`

### `get_QD`

```mojo
fn get_QD(self: _Self) -> UnsafePointer[Float64, origin_of(MutOrigin.external)]
```

**Args:**

- **self** (`_Self`)

**Returns:**

`UnsafePointer`

### `swap_index`

```mojo
fn swap_index(mut self: _Self, i: Int, j: Int)
```

**Args:**

- **self** (`_Self`)
- **i** (`Int`)
- **j** (`Int`)


