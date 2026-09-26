Mojo struct

# `IVec`

```mojo
@memory_only
struct IVec
```

## Fields

- **data** (`Pointer[Int, MutUntrackedOrigin]`)
- **n** (`Int`)
- **owns** (`Bool`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`, `Sized`

## Methods

### `__init__`

```mojo
fn def __init__(out self, n: Int)
```

**Args:**

- **n** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, data: Pointer[Int, MutUntrackedOrigin], n: Int)
```

**Args:**

- **data** (`Pointer[Int, MutUntrackedOrigin]`)
- **n** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

### `__deinit__`

```mojo
fn def __deinit__(deinit self)
```

**Args:**

- **self** (`Self`)

### `__getitem__`

```mojo
fn def __getitem__(self, i: Int) -> Int
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`Int`

### `__setitem__`

```mojo
fn def __setitem__(mut self, i: Int, v: Int)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **v** (`Int`)

### `__len__`

```mojo
fn def __len__(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `segment`

```mojo
fn def segment(self, start: Int, length: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **start** (`Int`)
- **length** (`Int`)

**Returns:**

`Self`


