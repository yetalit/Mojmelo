Mojo struct

# `Matrix`

```mojo
@memory_only
struct Matrix[Type: DType]
```

## Parameters

- **Type** (`DType`)

## Fields

- **data** (`UnsafePointer[Scalar[Type], MutAnyOrigin]`)
- **layout** (`Layout`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
def __init__(out self, shape: Tuple[Int, Int])
```

**Args:**

- **shape** (`Tuple`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
def __init__(out self, data: UnsafePointer[Scalar[Type], MutAnyOrigin], var layout: Layout)
```

**Args:**

- **data** (`UnsafePointer`)
- **layout** (`Layout`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
def __init__(out self, data: UnsafePointer[Scalar[Type], MutAnyOrigin], shape: Tuple[Int, Int])
```

**Args:**

- **data** (`UnsafePointer`)
- **shape** (`Tuple`)
- **self** (`Self`)

**Returns:**

`Self`

### `__getitem__`

```mojo
def __getitem__(ref self, i: Int, j: Int) -> ref[self_is_mut] Scalar[Type]
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

**Returns:**

`ref`

### `slice`

```mojo
def slice(self, i: Int, j: Int, ir: Int, jr: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)
- **ir** (`Int`)
- **jr** (`Int`)

**Returns:**

`Self`

### `shape`

```mojo
def shape[dim: Int](self) -> Int
```

**Parameters:**

- **dim** (`Int`)

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `stride`

```mojo
def stride[dim: Int](self) -> Int
```

**Parameters:**

- **dim** (`Int`)

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `rand`

```mojo
def rand(mut self)
```

**Args:**

- **self** (`Self`)

### `load`

```mojo
def load[width: Int, *, dim: Int](self, i: Int, j: Int) -> SIMD[Type, width]
```

**Parameters:**

- **width** (`Int`)
- **dim** (`Int`)

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

**Returns:**

`SIMD`

### `store`

```mojo
def store[width: Int, *, dim: Int](self, value: SIMD[Type, width], i: Int, j: Int)
```

**Parameters:**

- **width** (`Int`)
- **dim** (`Int`)

**Args:**

- **self** (`Self`)
- **value** (`SIMD`)
- **i** (`Int`)
- **j** (`Int`)

### `write_to`

```mojo
def write_to[W: Writer](self, mut writer: W)
```

**Parameters:**

- **W** (`Writer`)

**Args:**

- **self** (`Self`)
- **writer** (`W`)


