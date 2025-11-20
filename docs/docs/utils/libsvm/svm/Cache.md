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
- **head** (`UnsafePointer[head_t, origin_of(MutOrigin.external)]`)
- **lru_head** (`head_t`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

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
fn __del__(var self)
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
fn lru_insert(mut self, h: UnsafePointer[head_t, origin_of(MutOrigin.external)])
```

**Args:**

- **self** (`Self`)
- **h** (`UnsafePointer`)

### `get_data`

```mojo
fn get_data(mut self, index: Int, data: UnsafePointer[UnsafePointer[Float32, origin_of(MutOrigin.external)], MutAnyOrigin], var _len: Int) -> Int
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


