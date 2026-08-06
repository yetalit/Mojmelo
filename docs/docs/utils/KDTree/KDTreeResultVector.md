Mojo struct

# `KDTreeResultVector`

```mojo
@memory_only
struct KDTreeResultVector
```

## Implemented traits

`AnyType`, `Copyable`, `Deinitable`, `Movable`, `Sized`

## Methods

### `__init__`

```mojo
fn def __init__(out self)
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `__getitem__`

```mojo
fn def __getitem__(self, index: Int) -> KDTreeResult
```

**Args:**

- **self** (`Self`)
- **index** (`Int`)

**Returns:**

`KDTreeResult`

### `__setitem__`

```mojo
fn def __setitem__(mut self, index: Int, val: KDTreeResult)
```

**Args:**

- **self** (`Self`)
- **index** (`Int`)
- **val** (`KDTreeResult`)

### `__len__`

```mojo
fn def __len__(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `append_heap`

```mojo
fn def append_heap(mut self)
```

**Args:**

- **self** (`Self`)

### `append_element_and_heapify`

```mojo
fn def append_element_and_heapify(mut self, e: KDTreeResult)
```

**Args:**

- **self** (`Self`)
- **e** (`KDTreeResult`)

### `pop_heap`

```mojo
fn def pop_heap(mut self)
```

**Args:**

- **self** (`Self`)

### `max_value`

```mojo
fn def max_value(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

### `replace_maxpri_elt_return_new_maxpri`

```mojo
fn def replace_maxpri_elt_return_new_maxpri(mut self, e: KDTreeResult) -> Float32
```

**Args:**

- **self** (`Self`)
- **e** (`KDTreeResult`)

**Returns:**

`Float32`


