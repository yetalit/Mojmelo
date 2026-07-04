Mojo struct

# `Matrix`

```mojo
@memory_only
struct Matrix
```

Native matrix data structure.

## Aliases

- `simd_width = (4 * simd_width_of[DType.float32]()) if CompilationTarget.is_apple_silicon() else (2 * simd_width_of[DType.float32]())`

## Fields

- **height** (`Int`): The number of rows.
- **width** (`Int`): The number of columns.
- **size** (`Int`): The total size.
- **data** (`UnsafePointer[Float32, MutAnyOrigin]`): The pointer to the underlying data.
- **order** (`String`): The order of matrix: Row-major -> 'c'; Column-major -> 'f'.

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDeletable`, `Movable`, `Sized`, `Writable`

## Methods

### `__init__`

```mojo
fn def __init__[src: DType = DType.float32](out self, data: UnsafePointer[Scalar[src], MutAnyOrigin], height: Int, width: Int, order: String = "c")
```

**Parameters:**

- **src** (`DType`)

**Args:**

- **data** (`UnsafePointer[Scalar[src], MutAnyOrigin]`)
- **height** (`Int`)
- **width** (`Int`)
- **order** (`String`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, height: Int, width: Int, data: Optional[UnsafePointer[Float32, MutAnyOrigin]] = None, order: String = "c")
```

**Args:**

- **height** (`Int`)
- **width** (`Int`)
- **data** (`Optional[UnsafePointer[Float32, MutAnyOrigin]]`)
- **order** (`String`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, def_input: List[List[Float32]])
```

**Args:**

- **def_input** (`List[List[Float32]]`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __init__(out self, *, copy: Self)
```

**Args:**

- **copy** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn def __init__(out self, *, deinit take: Self)
```

**Args:**

- **take** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `__getitem__`

```mojo
fn def __getitem__(self, row: Int, column: Int) -> Float32
```

The pattern to access a single value: [row, column] .

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **column** (`Int`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn def __getitem__(self, row: Int) -> Self
```

The pattern to access a row: [row] .

**Args:**

- **self** (`Self`)
- **row** (`Int`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, row: Int, *, unsafe: Bool) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **unsafe** (`Bool`)

**Returns:**

`Self`

```mojo
fn def __getitem__(self, row: Int, offset: Bool, start_i: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **offset** (`Bool`)
- **start_i** (`Int`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, row: String, column: Int) -> Self
```

The pattern to access a column: ['', column] .

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, row: String, column: Int, *, unsafe: Bool) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **unsafe** (`Bool`)

**Returns:**

`Self`

```mojo
fn def __getitem__(self, offset: Bool, start_i: Int, column: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **offset** (`Bool`)
- **start_i** (`Int`)
- **column** (`Int`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, rows: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, row: String, columns: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, rows: List[Int]) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`List[Int]`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, rows: List[Int]) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`List[Int]`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, row: String, columns: List[Int]) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`List[Int]`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __getitem__(self, row: String, columns: List[Int]) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`List[Int]`)

**Returns:**

`Self`

**Raises:**

### `__setitem__`

```mojo
fn def __setitem__(mut self, row: Int, column: Int, val: Float32)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **column** (`Int`)
- **val** (`Float32`)

**Raises:**

```mojo
fn def __setitem__(mut self, row: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
fn def __setitem__(mut self, row: Int, val: Self, *, unsafe: Bool)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **val** (`Self`)
- **unsafe** (`Bool`)

```mojo
fn def __setitem__(mut self, row: Int, offset: Bool, start_i: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **offset** (`Bool`)
- **start_i** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
fn def __setitem__(mut self, row: String, column: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
fn def __setitem__(mut self, row: String, column: Int, val: Self, *, unsafe: Bool)
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **val** (`Self`)
- **unsafe** (`Bool`)

```mojo
fn def __setitem__(mut self, offset: Bool, start_i: Int, column: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **offset** (`Bool`)
- **start_i** (`Int`)
- **column** (`Int`)
- **val** (`Self`)

**Raises:**

### `__neg__`

```mojo
fn def __neg__(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `__lt__`

```mojo
fn def __lt__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List[Scalar[DType.bool]]`

### `__le__`

```mojo
fn def __le__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List[Scalar[DType.bool]]`

### `__eq__`

```mojo
fn def __eq__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List[Scalar[DType.bool]]`

```mojo
fn def __eq__(self, rhs: Self) -> Bool
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Bool`

### `__ne__`

```mojo
fn def __ne__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List[Scalar[DType.bool]]`

```mojo
fn def __ne__(self, rhs: Self) -> Bool
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Bool`

### `__gt__`

```mojo
fn def __gt__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List[Scalar[DType.bool]]`

### `__ge__`

```mojo
fn def __ge__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List[Scalar[DType.bool]]`

### `__add__`

```mojo
fn def __add__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __add__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__sub__`

```mojo
fn def __sub__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __sub__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__mul__`

```mojo
fn def __mul__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __mul__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__truediv__`

```mojo
fn def __truediv__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def __truediv__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__pow__`

```mojo
fn def __pow__(self, p: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **p** (`Int`)

**Returns:**

`Self`

### `__radd__`

```mojo
fn def __radd__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rsub__`

```mojo
fn def __rsub__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rmul__`

```mojo
fn def __rmul__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rtruediv__`

```mojo
fn def __rtruediv__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__iadd__`

```mojo
fn def __iadd__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn def __iadd__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__isub__`

```mojo
fn def __isub__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn def __isub__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__imul__`

```mojo
fn def __imul__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn def __imul__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__itruediv__`

```mojo
fn def __itruediv__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn def __itruediv__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__ipow__`

```mojo
fn def __ipow__(mut self, rhs: Int)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Int`)

### `load`

```mojo
fn def load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]
```

**Parameters:**

- **nelts** (`Int`)

**Args:**

- **self** (`Self`)
- **y** (`Int`)
- **x** (`Int`)

**Returns:**

`SIMD[DType.float32, nelts]`

### `store`

```mojo
fn def store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts])
```

**Parameters:**

- **nelts** (`Int`)

**Args:**

- **self** (`Self`)
- **y** (`Int`)
- **x** (`Int`)
- **val** (`SIMD[DType.float32, nelts]`)

### `load_columns`

```mojo
fn def load_columns(self, _range: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **_range** (`Int`)

**Returns:**

`Self`

**Raises:**

### `load_rows`

```mojo
fn def load_rows(self, _range: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **_range** (`Int`)

**Returns:**

`Self`

**Raises:**

### `get_per_row`

```mojo
fn def get_per_row(self, columns: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **columns** (`Self`)

**Returns:**

`Self`

**Raises:**

### `set_per_row`

```mojo
fn def set_per_row(mut self, columns: Self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **columns** (`Self`)
- **rhs** (`Self`)

**Raises:**

### `__len__`

```mojo
fn def __len__(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `ele_eq`

```mojo
fn def ele_eq(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List[Scalar[DType.bool]]`

### `ele_ne`

```mojo
fn def ele_ne(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List[Scalar[DType.bool]]`

### `ele_gt`

```mojo
fn def ele_gt(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List[Scalar[DType.bool]]`

### `ele_ge`

```mojo
fn def ele_ge(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List[Scalar[DType.bool]]`

### `ele_lt`

```mojo
fn def ele_lt(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List[Scalar[DType.bool]]`

### `ele_le`

```mojo
fn def ele_le(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List[Scalar[DType.bool]]`

### `ele_mul`

```mojo
fn def ele_mul(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

### `where`

```mojo
fn def where(self, cmp: List[Scalar[DType.bool]], _true: Float32, _false: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List[Scalar[DType.bool]]`)
- **_true** (`Float32`)
- **_false** (`Float32`)

**Returns:**

`Self`

```mojo
fn def where(self, cmp: List[Scalar[DType.bool]], _true: Self, _false: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List[Scalar[DType.bool]]`)
- **_true** (`Self`)
- **_false** (`Float32`)

**Returns:**

`Self`

```mojo
fn def where(self, cmp: List[Scalar[DType.bool]], _true: Float32, _false: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List[Scalar[DType.bool]]`)
- **_true** (`Float32`)
- **_false** (`Self`)

**Returns:**

`Self`

```mojo
fn def where(self, cmp: List[Scalar[DType.bool]], _true: Self, _false: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List[Scalar[DType.bool]]`)
- **_true** (`Self`)
- **_false** (`Self`)

**Returns:**

`Self`

### `C_transpose`

```mojo
fn def C_transpose(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `F_transpose`

```mojo
fn def F_transpose(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `T`

```mojo
fn def T(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `asorder`

```mojo
fn def asorder(self, order: String) -> Self
```

**Args:**

- **self** (`Self`)
- **order** (`String`)

**Returns:**

`Self`

### `cumsum`

```mojo
fn def cumsum(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `sum`

```mojo
fn def sum(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn def sum(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `mean`

```mojo
fn def mean(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn def mean(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `mean_weighted`

```mojo
fn def mean_weighted(self, weights: Self, size: Float32) -> Float32
```

**Args:**

- **self** (`Self`)
- **weights** (`Self`)
- **size** (`Float32`)

**Returns:**

`Float32`

**Raises:**

### `mean_slow`

```mojo
fn def mean_slow(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

### `mean_slow0`

```mojo
fn def mean_slow0(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `std`

```mojo
fn def std(self, correction: Bool = False) -> Float32
```

**Args:**

- **self** (`Self`)
- **correction** (`Bool`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn def std(self, _mean: Float32, correction: Bool = False) -> Float32
```

**Args:**

- **self** (`Self`)
- **_mean** (`Float32`)
- **correction** (`Bool`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn def std(self, axis: Int, correction: Bool = False) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)
- **correction** (`Bool`)

**Returns:**

`Self`

**Raises:**

```mojo
fn def std(self, axis: Int, _mean: Self, correction: Bool = False) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)
- **_mean** (`Self`)
- **correction** (`Bool`)

**Returns:**

`Self`

**Raises:**

### `std_slow`

```mojo
fn def std_slow(self, _mean: Float32) -> Float32
```

**Args:**

- **self** (`Self`)
- **_mean** (`Float32`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn def std_slow(self, axis: Int, _mean: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)
- **_mean** (`Self`)

**Returns:**

`Self`

**Raises:**

### `abs`

```mojo
fn def abs(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `log`

```mojo
fn def log(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `sqrt`

```mojo
fn def sqrt(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `exp`

```mojo
fn def exp(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `argmin`

```mojo
fn def argmin(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

```mojo
fn def argmin(self, axis: Int) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`List[Int]`

### `argmax`

```mojo
fn def argmax(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

```mojo
fn def argmax(self, axis: Int) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`List[Int]`

### `argmax_f`

```mojo
fn def argmax_f(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

### `argsort`

```mojo
fn def argsort[ascending: Bool = True](self) -> List[Int]
```

**Parameters:**

- **ascending** (`Bool`)

**Args:**

- **self** (`Self`)

**Returns:**

`List[Int]`

**Raises:**

### `argsort_inplace`

```mojo
fn def argsort_inplace[ascending: Bool = True](mut self, mut sorted_indices: List[Int])
```

**Parameters:**

- **ascending** (`Bool`)

**Args:**

- **self** (`Self`)
- **sorted_indices** (`List[Int]`)

**Raises:**

### `min`

```mojo
fn def min(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn def min(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `max`

```mojo
fn def max(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn def max(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `reshape`

```mojo
fn def reshape(self, height: Int, width: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **height** (`Int`)
- **width** (`Int`)

**Returns:**

`Self`

### `lu_factor`

```mojo
@staticmethod
fn def lu_factor(mut A, piv: UnsafePointer[Int, MutAnyOrigin], N: Int)
```

**Args:**

- **A** (`Self`)
- **piv** (`UnsafePointer[Int, MutAnyOrigin]`)
- **N** (`Int`)

**Raises:**

### `lu_solve`

```mojo
@staticmethod
fn def lu_solve(A, piv: UnsafePointer[Int, MutAnyOrigin], b: Self, mut x: Self, N: Int, Mi: Int)
```

**Args:**

- **A** (`Self`)
- **piv** (`UnsafePointer[Int, MutAnyOrigin]`)
- **b** (`Self`)
- **x** (`Self`)
- **N** (`Int`)
- **Mi** (`Int`)

**Raises:**

### `solve`

```mojo
@staticmethod
fn def solve(var A, b: Self) -> Self
```

**Args:**

- **A** (`Self`)
- **b** (`Self`)

**Returns:**

`Self`

**Raises:**

### `inv`

```mojo
fn def inv(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `eye`

```mojo
@staticmethod
fn def eye(var n: Int, order: String = "c") -> Self
```

**Args:**

- **n** (`Int`)
- **order** (`String`)

**Returns:**

`Self`

### `norm`

```mojo
fn def norm(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

### `outer`

```mojo
fn def outer(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

### `concatenate`

```mojo
fn def concatenate(self, rhs: Self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `bincount`

```mojo
fn def bincount(self) -> List[Int]
```

**Args:**

- **self** (`Self`)

**Returns:**

`List[Int]`

**Raises:**

```mojo
fn def bincount(self, weights: Self) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **weights** (`Self`)

**Returns:**

`List[Int]`

**Raises:**

### `unique`

```mojo
fn def unique(self) -> List[List[Int]]
```

**Args:**

- **self** (`Self`)

**Returns:**

`List[List[Int]]`

```mojo
fn def unique(self, weights: Self) -> List[List[Int]]
```

**Args:**

- **self** (`Self`)
- **weights** (`Self`)

**Returns:**

`List[List[Int]]`

### `is_uniquef`

```mojo
fn def is_uniquef(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `zeros`

```mojo
@staticmethod
fn def zeros(height: Int, width: Int, order: String = "c") -> Self
```

**Args:**

- **height** (`Int`)
- **width** (`Int`)
- **order** (`String`)

**Returns:**

`Self`

### `ones`

```mojo
@staticmethod
fn def ones(height: Int, width: Int, order: String = "c") -> Self
```

**Args:**

- **height** (`Int`)
- **width** (`Int`)
- **order** (`String`)

**Returns:**

`Self`

### `full`

```mojo
@staticmethod
fn def full(height: Int, width: Int, val: Float32, order: String = "c") -> Self
```

**Args:**

- **height** (`Int`)
- **width** (`Int`)
- **val** (`Float32`)
- **order** (`String`)

**Returns:**

`Self`

### `fill_zero`

```mojo
fn def fill_zero(self)
```

**Args:**

- **self** (`Self`)

### `fill`

```mojo
fn def fill(self, val: Float32)
```

**Args:**

- **self** (`Self`)
- **val** (`Float32`)

### `random`

```mojo
@staticmethod
fn def random(height: Int, width: Int, order: String = "c") -> Self
```

**Args:**

- **height** (`Int`)
- **width** (`Int`)
- **order** (`String`)

**Returns:**

`Self`

### `rand_choice`

```mojo
@staticmethod
fn def rand_choice(arang: Int, size: Int, replace: Bool = True, seed: Bool = True) -> List[Int]
```

**Args:**

- **arang** (`Int`)
- **size** (`Int`)
- **replace** (`Bool`)
- **seed** (`Bool`)

**Returns:**

`List[Int]`

**Raises:**

### `linspace`

```mojo
@staticmethod
fn def linspace(start: Float32, stop: Float32, num: Int, order: String = "c") -> Self
```

**Args:**

- **start** (`Float32`)
- **stop** (`Float32`)
- **num** (`Int`)
- **order** (`String`)

**Returns:**

`Self`

**Raises:**

### `from_numpy`

```mojo
@staticmethod
fn def from_numpy(np_arr: PythonObject, order: String = "c") -> Self
```

Initialize a matrix from a numpy array.

**Args:**

- **np_arr** (`PythonObject`)
- **order** (`String`)

**Returns:**

`Self`: The matrix.

**Raises:**

### `to_numpy`

```mojo
fn def to_numpy(self) -> PythonObject
```

Converts the matrix to a numpy array.

**Args:**

- **self** (`Self`)

**Returns:**

`PythonObject`: The numpy array.

**Raises:**

### `cast_ptr`

```mojo
fn def cast_ptr[des: DType](self) -> UnsafePointer[Scalar[des], MutUntrackedOrigin]
```

**Parameters:**

- **des** (`DType`)

**Args:**

- **self** (`Self`)

**Returns:**

`UnsafePointer[Scalar[des], MutUntrackedOrigin]`

### `write_to`

```mojo
fn def write_to[W: Writer](self, mut writer: W)
```

**Parameters:**

- **W** (`Writer`)

**Args:**

- **self** (`Self`)
- **writer** (`W`)

### `__str__`

```mojo
fn def __str__(self) -> String
```

**Args:**

- **self** (`Self`)

**Returns:**

`String`


