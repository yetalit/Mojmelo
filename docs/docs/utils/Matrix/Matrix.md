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

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Movable`, `Sized`, `Writable`

## Methods

### `__init__`

```mojo
def __init__[src: DType = DType.float32](out self, data: UnsafePointer[Scalar[src], MutAnyOrigin], height: Int, width: Int, order: String = "c")
```

**Parameters:**

- **src** (`DType`)

**Args:**

- **data** (`UnsafePointer`)
- **height** (`Int`)
- **width** (`Int`)
- **order** (`String`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
def __init__(out self, height: Int, width: Int, data: UnsafePointer[Float32, MutAnyOrigin] = UnsafePointer(), order: String = "c")
```

**Args:**

- **height** (`Int`)
- **width** (`Int`)
- **data** (`UnsafePointer`)
- **order** (`String`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
def __init__(out self, def_input: List[List[Float32]])
```

**Args:**

- **def_input** (`List`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __init__(out self, *, copy: Self)
```

**Args:**

- **copy** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
def __init__(out self, *, deinit take: Self)
```

**Args:**

- **take** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
def __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `__getitem__`

```mojo
def __getitem__(self, row: Int, column: Int) -> Float32
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
def __getitem__(self, row: Int) -> Self
```

The pattern to access a row: [row] .

**Args:**

- **self** (`Self`)
- **row** (`Int`)

**Returns:**

`Self`

**Raises:**

```mojo
def __getitem__(self, row: Int, *, unsafe: Bool) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **unsafe** (`Bool`)

**Returns:**

`Self`

```mojo
def __getitem__(self, row: Int, offset: Bool, start_i: Int) -> Self
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
def __getitem__(self, row: String, column: Int) -> Self
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
def __getitem__(self, row: String, column: Int, *, unsafe: Bool) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **unsafe** (`Bool`)

**Returns:**

`Self`

```mojo
def __getitem__(self, offset: Bool, start_i: Int, column: Int) -> Self
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
def __getitem__(self, rows: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __getitem__(self, row: String, columns: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __getitem__(self, rows: List[Int]) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`List`)

**Returns:**

`Self`

**Raises:**

```mojo
def __getitem__(self, rows: List[Scalar[DType.int]]) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`List`)

**Returns:**

`Self`

**Raises:**

```mojo
def __getitem__(self, row: String, columns: List[Int]) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`List`)

**Returns:**

`Self`

**Raises:**

```mojo
def __getitem__(self, row: String, columns: List[Scalar[DType.int]]) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`List`)

**Returns:**

`Self`

**Raises:**

### `__setitem__`

```mojo
def __setitem__(mut self, row: Int, column: Int, val: Float32)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **column** (`Int`)
- **val** (`Float32`)

**Raises:**

```mojo
def __setitem__(mut self, row: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
def __setitem__(mut self, row: Int, val: Self, *, unsafe: Bool)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **val** (`Self`)
- **unsafe** (`Bool`)

```mojo
def __setitem__(mut self, row: Int, offset: Bool, start_i: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **offset** (`Bool`)
- **start_i** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
def __setitem__(mut self, row: String, column: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
def __setitem__(mut self, row: String, column: Int, val: Self, *, unsafe: Bool)
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **val** (`Self`)
- **unsafe** (`Bool`)

```mojo
def __setitem__(mut self, offset: Bool, start_i: Int, column: Int, val: Self)
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
def __neg__(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `__lt__`

```mojo
def __lt__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

### `__le__`

```mojo
def __le__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

### `__eq__`

```mojo
def __eq__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

```mojo
def __eq__(self, rhs: Self) -> Bool
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Bool`

### `__ne__`

```mojo
def __ne__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

```mojo
def __ne__(self, rhs: Self) -> Bool
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Bool`

### `__gt__`

```mojo
def __gt__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

### `__ge__`

```mojo
def __ge__(self, rhs: Float32) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

### `__add__`

```mojo
def __add__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __add__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__sub__`

```mojo
def __sub__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __sub__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__mul__`

```mojo
def __mul__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __mul__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__truediv__`

```mojo
def __truediv__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
def __truediv__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__pow__`

```mojo
def __pow__(self, p: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **p** (`Int`)

**Returns:**

`Self`

### `__radd__`

```mojo
def __radd__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rsub__`

```mojo
def __rsub__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rmul__`

```mojo
def __rmul__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rtruediv__`

```mojo
def __rtruediv__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__iadd__`

```mojo
def __iadd__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
def __iadd__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__isub__`

```mojo
def __isub__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
def __isub__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__imul__`

```mojo
def __imul__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
def __imul__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__itruediv__`

```mojo
def __itruediv__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
def __itruediv__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__ipow__`

```mojo
def __ipow__(mut self, rhs: Int)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Int`)

### `load`

```mojo
def load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]
```

**Parameters:**

- **nelts** (`Int`)

**Args:**

- **self** (`Self`)
- **y** (`Int`)
- **x** (`Int`)

**Returns:**

`SIMD`

### `store`

```mojo
def store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts])
```

**Parameters:**

- **nelts** (`Int`)

**Args:**

- **self** (`Self`)
- **y** (`Int`)
- **x** (`Int`)
- **val** (`SIMD`)

### `load_columns`

```mojo
def load_columns(self, _range: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **_range** (`Int`)

**Returns:**

`Self`

**Raises:**

### `load_rows`

```mojo
def load_rows(self, _range: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **_range** (`Int`)

**Returns:**

`Self`

**Raises:**

### `get_per_row`

```mojo
def get_per_row(self, columns: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **columns** (`Self`)

**Returns:**

`Self`

**Raises:**

### `set_per_row`

```mojo
def set_per_row(mut self, columns: Self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **columns** (`Self`)
- **rhs** (`Self`)

**Raises:**

### `__len__`

```mojo
def __len__(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `ele_eq`

```mojo
def ele_eq(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List`

### `ele_ne`

```mojo
def ele_ne(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List`

### `ele_gt`

```mojo
def ele_gt(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List`

### `ele_ge`

```mojo
def ele_ge(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List`

### `ele_lt`

```mojo
def ele_lt(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List`

### `ele_le`

```mojo
def ele_le(self, rhs: Self) -> List[Scalar[DType.bool]]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`List`

### `ele_mul`

```mojo
def ele_mul(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

### `where`

```mojo
def where(self, cmp: List[Scalar[DType.bool]], _true: Float32, _false: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)
- **_true** (`Float32`)
- **_false** (`Float32`)

**Returns:**

`Self`

```mojo
def where(self, cmp: List[Scalar[DType.bool]], _true: Self, _false: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)
- **_true** (`Self`)
- **_false** (`Float32`)

**Returns:**

`Self`

```mojo
def where(self, cmp: List[Scalar[DType.bool]], _true: Float32, _false: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)
- **_true** (`Float32`)
- **_false** (`Self`)

**Returns:**

`Self`

```mojo
def where(self, cmp: List[Scalar[DType.bool]], _true: Self, _false: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)
- **_true** (`Self`)
- **_false** (`Self`)

**Returns:**

`Self`

### `argwhere_l`

```mojo
def argwhere_l(self, cmp: List[Scalar[DType.bool]]) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)

**Returns:**

`List`

### `C_transpose`

```mojo
def C_transpose(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `F_transpose`

```mojo
def F_transpose(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `T`

```mojo
def T(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `asorder`

```mojo
def asorder(self, order: String) -> Self
```

**Args:**

- **self** (`Self`)
- **order** (`String`)

**Returns:**

`Self`

### `cumsum`

```mojo
def cumsum(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `sum`

```mojo
def sum(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
def sum(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `mean`

```mojo
def mean(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
def mean(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `mean_weighted`

```mojo
def mean_weighted(self, weights: Self, size: Float32) -> Float32
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
def mean_slow(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

### `mean_slow0`

```mojo
def mean_slow0(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `std`

```mojo
def std(self, correction: Bool = False) -> Float32
```

**Args:**

- **self** (`Self`)
- **correction** (`Bool`)

**Returns:**

`Float32`

**Raises:**

```mojo
def std(self, _mean: Float32, correction: Bool = False) -> Float32
```

**Args:**

- **self** (`Self`)
- **_mean** (`Float32`)
- **correction** (`Bool`)

**Returns:**

`Float32`

**Raises:**

```mojo
def std(self, axis: Int, correction: Bool = False) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)
- **correction** (`Bool`)

**Returns:**

`Self`

**Raises:**

```mojo
def std(self, axis: Int, _mean: Self, correction: Bool = False) -> Self
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
def std_slow(self, _mean: Float32) -> Float32
```

**Args:**

- **self** (`Self`)
- **_mean** (`Float32`)

**Returns:**

`Float32`

**Raises:**

```mojo
def std_slow(self, axis: Int, _mean: Self) -> Self
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
def abs(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `log`

```mojo
def log(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `sqrt`

```mojo
def sqrt(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `exp`

```mojo
def exp(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `argmin`

```mojo
def argmin(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

```mojo
def argmin(self, axis: Int) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`List`

### `argmax`

```mojo
def argmax(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

```mojo
def argmax(self, axis: Int) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`List`

### `argmax_f`

```mojo
def argmax_f(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

### `argsort`

```mojo
def argsort[ascending: Bool = True](self) -> List[Scalar[DType.int]]
```

**Parameters:**

- **ascending** (`Bool`)

**Args:**

- **self** (`Self`)

**Returns:**

`List`

**Raises:**

### `argsort_inplace`

```mojo
def argsort_inplace[ascending: Bool = True](mut self) -> List[Scalar[DType.int]]
```

**Parameters:**

- **ascending** (`Bool`)

**Args:**

- **self** (`Self`)

**Returns:**

`List`

**Raises:**

### `min`

```mojo
def min(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
def min(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `max`

```mojo
def max(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
def max(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `reshape`

```mojo
def reshape(self, height: Int, width: Int) -> Self
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
def lu_factor(mut A, piv: UnsafePointer[Int, MutAnyOrigin], N: Int)
```

**Args:**

- **A** (`Self`)
- **piv** (`UnsafePointer`)
- **N** (`Int`)

**Raises:**

### `lu_solve`

```mojo
@staticmethod
def lu_solve(A, piv: UnsafePointer[Int, MutAnyOrigin], b: Self, mut x: Self, N: Int, Mi: Int)
```

**Args:**

- **A** (`Self`)
- **piv** (`UnsafePointer`)
- **b** (`Self`)
- **x** (`Self`)
- **N** (`Int`)
- **Mi** (`Int`)

**Raises:**

### `solve`

```mojo
@staticmethod
def solve(var A, b: Self) -> Self
```

**Args:**

- **A** (`Self`)
- **b** (`Self`)

**Returns:**

`Self`

**Raises:**

### `inv`

```mojo
def inv(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `eye`

```mojo
@staticmethod
def eye(n: Int, order: String = "c") -> Self
```

**Args:**

- **n** (`Int`)
- **order** (`String`)

**Returns:**

`Self`

### `norm`

```mojo
def norm(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

### `outer`

```mojo
def outer(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

### `concatenate`

```mojo
def concatenate(self, rhs: Self, axis: Int) -> Self
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
def bincount(self) -> List[Int]
```

**Args:**

- **self** (`Self`)

**Returns:**

`List`

**Raises:**

```mojo
def bincount(self, weights: Self) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **weights** (`Self`)

**Returns:**

`List`

**Raises:**

### `unique`

```mojo
def unique(self) -> List[List[Int]]
```

**Args:**

- **self** (`Self`)

**Returns:**

`List`

```mojo
def unique(self, weights: Self) -> List[List[Int]]
```

**Args:**

- **self** (`Self`)
- **weights** (`Self`)

**Returns:**

`List`

### `is_uniquef`

```mojo
def is_uniquef(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `zeros`

```mojo
@staticmethod
def zeros(height: Int, width: Int, order: String = "c") -> Self
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
def ones(height: Int, width: Int, order: String = "c") -> Self
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
def full(height: Int, width: Int, val: Float32, order: String = "c") -> Self
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
def fill_zero(self)
```

**Args:**

- **self** (`Self`)

### `fill`

```mojo
def fill(self, val: Float32)
```

**Args:**

- **self** (`Self`)
- **val** (`Float32`)

### `random`

```mojo
@staticmethod
def random(height: Int, width: Int, order: String = "c") -> Self
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
def rand_choice(arang: Int, size: Int, replace: Bool = True, seed: Bool = True) -> List[Scalar[DType.int]]
```

**Args:**

- **arang** (`Int`)
- **size** (`Int`)
- **replace** (`Bool`)
- **seed** (`Bool`)

**Returns:**

`List`

**Raises:**

### `linspace`

```mojo
@staticmethod
def linspace(start: Float32, stop: Float32, num: Int, order: String = "c") -> Self
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
def from_numpy(np_arr: PythonObject, order: String = "c") -> Self
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
def to_numpy(self) -> PythonObject
```

Converts the matrix to a numpy array.

**Args:**

- **self** (`Self`)

**Returns:**

`PythonObject`: The numpy array.

**Raises:**

### `cast_ptr`

```mojo
def cast_ptr[des: DType](self) -> UnsafePointer[Scalar[des], MutExternalOrigin]
```

**Parameters:**

- **des** (`DType`)

**Args:**

- **self** (`Self`)

**Returns:**

`UnsafePointer`

### `write_to`

```mojo
def write_to[W: Writer](self, mut writer: W)
```

**Parameters:**

- **W** (`Writer`)

**Args:**

- **self** (`Self`)
- **writer** (`W`)

### `__str__`

```mojo
def __str__(self) -> String
```

**Args:**

- **self** (`Self`)

**Returns:**

`String`


