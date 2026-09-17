Mojo struct

# `Vec`

```mojo
@memory_only
struct Vec
```

## Fields

- **data** (`Pointer[Float64, MutUntrackedOrigin]`)
- **n** (`Int`)
- **stride** (`Int`)
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
fn def __init__(out self, data: Pointer[Float64, MutUntrackedOrigin], n: Int, stride: Int)
```

**Args:**

- **data** (`Pointer[Float64, MutUntrackedOrigin]`)
- **n** (`Int`)
- **stride** (`Int`)
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
fn def __getitem__(self, i: Int) -> RealScalar
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)

**Returns:**

`RealScalar`

### `__setitem__`

```mojo
fn def __setitem__(mut self, i: Int, v: Float64)
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **v** (`Float64`)

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

### `head`

```mojo
fn def head(self, length: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **length** (`Int`)

**Returns:**

`Self`

### `tail`

```mojo
fn def tail(self, length: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **length** (`Int`)

**Returns:**

`Self`

### `copyFrom`

```mojo
fn def copyFrom(self, other: Self)
```

**Args:**

- **self** (`Self`)
- **other** (`Self`)

### `setZero`

```mojo
fn def setZero(self)
```

**Args:**

- **self** (`Self`)

### `cwiseAbsMax`

```mojo
fn def cwiseAbsMax(self) -> RealScalar
```

**Args:**

- **self** (`Self`)

**Returns:**

`RealScalar`

### `norm`

```mojo
fn def norm(self) -> RealScalar
```

**Args:**

- **self** (`Self`)

**Returns:**

`RealScalar`

### `reverseInPlace`

```mojo
fn def reverseInPlace(mut self)
```

**Args:**

- **self** (`Self`)

### `stableNormalize`

```mojo
fn def stableNormalize(mut self)
```

**Args:**

- **self** (`Self`)

### `Unit`

```mojo
@staticmethod
fn def Unit(n: Int, k: Int) -> Self
```

**Args:**

- **n** (`Int`)
- **k** (`Int`)

**Returns:**

`Self`


