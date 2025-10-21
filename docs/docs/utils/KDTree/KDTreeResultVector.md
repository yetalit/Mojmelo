Mojo struct

# `KDTreeResultVector`

```mojo
@memory_only
struct KDTreeResultVector
```

## Aliases

- `__del__is_trivial = False`
- `__moveinit__is_trivial = True`
- `__copyinit__is_trivial = False`

## Implemented traits

`AnyType`, `Copyable`, `Movable`, `Sized`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self)
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `__getitem__`

```mojo
fn __getitem__(self, index: Int) -> KDTreeResult
```

**Args:**

- **self** (`Self`)
- **index** (`Int`)

**Returns:**

`KDTreeResult`

### `__setitem__`

```mojo
fn __setitem__(mut self, index: Int, val: KDTreeResult)
```

**Args:**

- **self** (`Self`)
- **index** (`Int`)
- **val** (`KDTreeResult`)

### `__len__`

```mojo
fn __len__(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `append_heap`

```mojo
fn append_heap(mut self)
```

**Args:**

- **self** (`Self`)

### `append_element_and_heapify`

```mojo
fn append_element_and_heapify(mut self, e: KDTreeResult)
```

**Args:**

- **self** (`Self`)
- **e** (`KDTreeResult`)

### `pop_heap`

```mojo
fn pop_heap(mut self)
```

**Args:**

- **self** (`Self`)

### `max_value`

```mojo
fn max_value(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

### `replace_maxpri_elt_return_new_maxpri`

```mojo
fn replace_maxpri_elt_return_new_maxpri(mut self, e: KDTreeResult) -> Float32
```

**Args:**

- **self** (`Self`)
- **e** (`KDTreeResult`)

**Returns:**

`Float32`


