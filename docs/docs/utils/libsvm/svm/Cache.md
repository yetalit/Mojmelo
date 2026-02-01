Mojo struct

# `Cache`

```mojo
@memory_only
struct Cache
```

## Aliases

- `__del__is_trivial = False`

## Fields

- **l** (`Int`)
- **size** (`UInt`)
- **head** (`UnsafePointer[head_t, MutExternalOrigin]`)
- **lru_head** (`head_t`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, l_: Int, size_: Scalar[DType.uindex])
```

**Args:**

- **l_** (`Int`)
- **size_** (`Scalar`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `lru_delete`

```mojo
fn lru_delete(self, h: UnsafePointer[head_t, MutAnyOrigin])
```

**Args:**

- **self** (`Self`)
- **h** (`UnsafePointer`)

### `lru_insert`

```mojo
fn lru_insert(mut self, h: UnsafePointer[head_t, MutExternalOrigin])
```

**Args:**

- **self** (`Self`)
- **h** (`UnsafePointer`)

### `get_data`

```mojo
fn get_data(mut self, index: Int, data: UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutAnyOrigin], var _len: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **index** (`Int`)
- **data** (`UnsafePointer`)
- **_len** (`Int`)

**Returns:**

`Int`

### `swap_index`

```mojo
fn swap_index(mut self, var i: Int, var j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)


