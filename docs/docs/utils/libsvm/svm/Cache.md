Mojo struct

# `Cache`

```mojo
@memory_only
struct Cache
```

## Fields

- **l** (`Int`)
- **size** (`UInt`)
- **head** (`Optional[UnsafePointer[head_t, MutExternalOrigin]]`)
- **lru_head** (`head_t`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, l_: Int, size_: UInt)
```

**Args:**

- **l_** (`Int`)
- **size_** (`UInt`)
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
fn lru_delete(self, h: Optional[UnsafePointer[head_t, MutAnyOrigin]])
```

**Args:**

- **self** (`Self`)
- **h** (`Optional[UnsafePointer[head_t, MutAnyOrigin]]`)

### `lru_insert`

```mojo
fn lru_insert(mut self, h: Optional[UnsafePointer[head_t, MutExternalOrigin]])
```

**Args:**

- **self** (`Self`)
- **h** (`Optional[UnsafePointer[head_t, MutExternalOrigin]]`)

### `get_data`

```mojo
fn get_data(mut self, index: Int, data: Optional[UnsafePointer[Optional[UnsafePointer[Float32, MutExternalOrigin]], MutAnyOrigin]], var _len: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **index** (`Int`)
- **data** (`Optional[UnsafePointer[Optional[UnsafePointer[Float32, MutExternalOrigin]], MutAnyOrigin]]`)
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


