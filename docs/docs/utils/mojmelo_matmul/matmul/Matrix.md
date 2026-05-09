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
fn __init__(out self, shape: Tuple[Int, Int])
```

**Args:**

- **shape** (`Tuple[Int, Int]`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn __init__(out self, data: UnsafePointer[Scalar[Type], MutAnyOrigin], var layout: Layout)
```

**Args:**

- **data** (`UnsafePointer[Scalar[Type], MutAnyOrigin]`)
- **layout** (`Layout`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn __init__(out self, data: UnsafePointer[Scalar[Type], MutAnyOrigin], shape: Tuple[Int, Int])
```

**Args:**

- **data** (`UnsafePointer[Scalar[Type], MutAnyOrigin]`)
- **shape** (`Tuple[Int, Int]`)
- **self** (`Self`)

**Returns:**

`Self`

### `__getitem__`

```mojo
fn __getitem__(ref self, i: Int, j: Int) -> ref[self_is_mut] Scalar[Type]
```

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

**Returns:**

`ref[self_is_mut] Scalar[Type]`

### `slice`

```mojo
fn slice(self, i: Int, j: Int, ir: Int, jr: Int) -> Self
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
fn shape[dim: Int](self) -> Int
```

**Parameters:**

- **dim** (`Int`)

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `stride`

```mojo
fn stride[dim: Int](self) -> Int
```

**Parameters:**

- **dim** (`Int`)

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `rand`

```mojo
fn rand(mut self)
```

**Args:**

- **self** (`Self`)

### `load`

```mojo
fn load[width: Int, *, dim: Int](self, i: Int, j: Int) -> SIMD[Type, width]
```

**Parameters:**

- **width** (`Int`)
- **dim** (`Int`)

**Args:**

- **self** (`Self`)
- **i** (`Int`)
- **j** (`Int`)

**Returns:**

`SIMD[Type, width]`

### `store`

```mojo
fn store[width: Int, *, dim: Int](self, value: SIMD[Type, width], i: Int, j: Int)
```

**Parameters:**

- **width** (`Int`)
- **dim** (`Int`)

**Args:**

- **self** (`Self`)
- **value** (`SIMD[Type, width]`)
- **i** (`Int`)
- **j** (`Int`)

### `write_to`

```mojo
fn write_to[W: Writer](self, mut writer: W)
```

**Parameters:**

- **W** (`Writer`)

**Args:**

- **self** (`Self`)
- **writer** (`W`)


