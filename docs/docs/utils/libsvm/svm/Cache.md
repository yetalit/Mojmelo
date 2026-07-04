Mojo struct

# `Cache`

```mojo
@memory_only
struct Cache
```

## Fields

- **l** (`Int`)
- **size** (`UInt`)
- **head** (`Optional[UnsafePointer[head_t, MutUntrackedOrigin]]`)
- **lru_head** (`head_t`)

## Implemented traits

`AnyType`, `ImplicitlyDeletable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, l_: Int, size_: UInt)
```

**Args:**

- **l_** (`Int`)
- **size_** (`UInt`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `lru_delete`

```mojo
fn def lru_delete(self, h: UnsafePointer[head_t, MutAnyOrigin])
```

**Args:**

- **self** (`Self`)
- **h** (`UnsafePointer[head_t, MutAnyOrigin]`)

### `lru_insert`

```mojo
fn def lru_insert(mut self, h: UnsafePointer[head_t, MutUntrackedOrigin])
```

**Args:**

- **self** (`Self`)
- **h** (`UnsafePointer[head_t, MutUntrackedOrigin]`)

### `get_data`

```mojo
fn def get_data(mut self, index: Int, data: Optional[UnsafePointer[Optional[UnsafePointer[Float32, MutUntrackedOrigin]], MutAnyOrigin]], var _len: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **index** (`Int`)
- **data** (`Optional[UnsafePointer[Optional[UnsafePointer[Float32, MutUntrackedOrigin]], MutAnyOrigin]]`)
- **_len** (`Int`)

**Returns:**

`Int`

### `swap_index`

```mojo
fn def swap_index(mut self, var i: Int, var j: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)


