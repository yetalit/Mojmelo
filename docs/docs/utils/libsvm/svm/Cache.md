Mojo struct

# `Cache`

```mojo
@memory_only
struct Cache
```

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
def __init__(out self, l_: Int, size_: UInt)
```

**Args:**

- **l_** (`Int`)
- **size_** (`UInt`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `lru_delete`

```mojo
def lru_delete(self, h: UnsafePointer[head_t, MutAnyOrigin])
```

**Args:**

- **self** (`Self`)
- **h** (`UnsafePointer`)

### `lru_insert`

```mojo
def lru_insert(mut self, h: UnsafePointer[head_t, MutExternalOrigin])
```

**Args:**

- **self** (`Self`)
- **h** (`UnsafePointer`)

### `get_data`

```mojo
def get_data(mut self, index: Int, data: UnsafePointer[UnsafePointer[Float32, MutExternalOrigin], MutAnyOrigin], var _len: Int) -> Int
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
def swap_index(mut self, var i: Int, var j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)


