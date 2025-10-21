Mojo struct

# `Matrix`

```mojo
@memory_only
struct Matrix
```

Native matrix data structure.

## Aliases

- `simd_width = (4 * simd_width_of[DType.float32]()) if CompilationTarget.is_apple_silicon[_current_target()]() else (2 * simd_width_of[DType.float32]())`
- `__del__is_trivial = False`
- `__moveinit__is_trivial = False`
- `__copyinit__is_trivial = False`

## Fields

- **height** (`Int`): The number of rows.
- **width** (`Int`): The number of columns.
- **size** (`Int`): The total size.
- **data** (`UnsafePointer[Float32]`): The pointer to the underlying data.
- **order** (`String`): The order of matrix: Row-major -> 'c'; Column-major -> 'f'.

## Implemented traits

`AnyType`, `Copyable`, `ImplicitlyCopyable`, `Movable`, `Sized`, `Stringable`, `UnknownDestructibility`, `Writable`

## Methods

### `__init__`

```mojo
fn __init__[src: DType = DType.float32](out self, data: UnsafePointer[Scalar[src]], height: Int, width: Int, order: String = "c")
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
fn __init__(out self, height: Int, width: Int, data: UnsafePointer[Float32] = UnsafePointer[Float32, AddressSpace(0), True, MutableAnyOrigin](), order: String = "c")
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
fn __init__(out self, def_input: List[List[Float32]])
```

**Args:**

- **def_input** (`List`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__copyinit__`

```mojo
@staticmethod
fn __copyinit__(out self, other: Self)
```

**Args:**

- **other** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

### `__moveinit__`

```mojo
@staticmethod
fn __moveinit__(out self, var existing: Self)
```

**Args:**

- **existing** (`Self`)
- **self** (`Self`)

**Returns:**

`Self`

### `__del__`

```mojo
fn __del__(var self)
```

**Args:**

- **self** (`Self`)

### `__getitem__`

```mojo
fn __getitem__(self, row: Int, column: Int) -> Float32
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
fn __getitem__(self, row: Int) -> Self
```

The pattern to access a row: [row] .

**Args:**

- **self** (`Self`)
- **row** (`Int`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __getitem__(self, row: Int, *, unsafe: Bool) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **unsafe** (`Bool`)

**Returns:**

`Self`

```mojo
fn __getitem__(self, row: Int, offset: Bool, start_i: Int) -> Self
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
fn __getitem__(self, row: String, column: Int) -> Self
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
fn __getitem__(self, row: String, column: Int, *, unsafe: Bool) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **unsafe** (`Bool`)

**Returns:**

`Self`

```mojo
fn __getitem__(self, offset: Bool, start_i: Int, column: Int) -> Self
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
fn __getitem__(self, rows: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __getitem__(self, row: String, columns: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __getitem__(self, rows: List[Int]) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`List`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __getitem__(self, rows: List[Scalar[DType.index]]) -> Self
```

**Args:**

- **self** (`Self`)
- **rows** (`List`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __getitem__(self, row: String, columns: List[Int]) -> Self
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`List`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __getitem__(self, row: String, columns: List[Scalar[DType.index]]) -> Self
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
fn __setitem__(mut self, row: Int, column: Int, val: Float32)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **column** (`Int`)
- **val** (`Float32`)

**Raises:**

```mojo
fn __setitem__(mut self, row: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
fn __setitem__(mut self, row: Int, val: Self, *, unsafe: Bool)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **val** (`Self`)
- **unsafe** (`Bool`)

```mojo
fn __setitem__(mut self, row: Int, offset: Bool, start_i: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`Int`)
- **offset** (`Bool`)
- **start_i** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
fn __setitem__(mut self, row: String, column: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
fn __setitem__(mut self, row: String, column: Int, val: Self, *, unsafe: Bool)
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **column** (`Int`)
- **val** (`Self`)
- **unsafe** (`Bool`)

```mojo
fn __setitem__(mut self, offset: Bool, start_i: Int, column: Int, val: Self)
```

**Args:**

- **self** (`Self`)
- **offset** (`Bool`)
- **start_i** (`Int`)
- **column** (`Int`)
- **val** (`Self`)

**Raises:**

```mojo
fn __setitem__(mut self, rows: Self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rows** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn __setitem__(mut self, row: String, columns: Self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **row** (`String`)
- **columns** (`Self`)
- **rhs** (`Self`)

**Raises:**

### `__neg__`

```mojo
fn __neg__(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `__lt__`

```mojo
fn __lt__(self, rhs: Float32) -> List[Bool]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

### `__le__`

```mojo
fn __le__(self, rhs: Float32) -> List[Bool]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

### `__eq__`

```mojo
fn __eq__(self, rhs: Float32) -> List[Bool]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

```mojo
fn __eq__(self, rhs: Self) -> Bool
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Bool`

### `__ne__`

```mojo
fn __ne__(self, rhs: Float32) -> List[Bool]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

```mojo
fn __ne__(self, rhs: Self) -> Bool
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Bool`

### `__gt__`

```mojo
fn __gt__(self, rhs: Float32) -> List[Bool]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

### `__ge__`

```mojo
fn __ge__(self, rhs: Float32) -> List[Bool]
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`List`

### `__add__`

```mojo
fn __add__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __add__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__sub__`

```mojo
fn __sub__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __sub__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__mul__`

```mojo
fn __mul__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __mul__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__truediv__`

```mojo
fn __truediv__(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

```mojo
fn __truediv__(self, rhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

**Returns:**

`Self`

### `__pow__`

```mojo
fn __pow__(self, p: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **p** (`Int`)

**Returns:**

`Self`

### `__radd__`

```mojo
fn __radd__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rsub__`

```mojo
fn __rsub__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rmul__`

```mojo
fn __rmul__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__rtruediv__`

```mojo
fn __rtruediv__(self, lhs: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **lhs** (`Float32`)

**Returns:**

`Self`

### `__iadd__`

```mojo
fn __iadd__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn __iadd__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__isub__`

```mojo
fn __isub__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn __isub__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__imul__`

```mojo
fn __imul__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn __imul__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__itruediv__`

```mojo
fn __itruediv__(mut self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Raises:**

```mojo
fn __itruediv__(mut self, rhs: Float32)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Float32`)

### `__ipow__`

```mojo
fn __ipow__(mut self, rhs: Int)
```

**Args:**

- **self** (`Self`)
- **rhs** (`Int`)

### `load`

```mojo
fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]
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
fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts])
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
fn load_columns(self, _range: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **_range** (`Int`)

**Returns:**

`Self`

**Raises:**

### `load_rows`

```mojo
fn load_rows(self, _range: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **_range** (`Int`)

**Returns:**

`Self`

**Raises:**

### `get_per_row`

```mojo
fn get_per_row(self, columns: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **columns** (`Self`)

**Returns:**

`Self`

**Raises:**

### `set_per_row`

```mojo
fn set_per_row(mut self, columns: Self, rhs: Self)
```

**Args:**

- **self** (`Self`)
- **columns** (`Self`)
- **rhs** (`Self`)

**Raises:**

### `__len__`

```mojo
fn __len__(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `ele_mul`

```mojo
fn ele_mul(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

### `where`

```mojo
fn where(self, cmp: List[Bool], _true: Float32, _false: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)
- **_true** (`Float32`)
- **_false** (`Float32`)

**Returns:**

`Self`

```mojo
fn where(self, cmp: List[Bool], _true: Self, _false: Float32) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)
- **_true** (`Self`)
- **_false** (`Float32`)

**Returns:**

`Self`

```mojo
fn where(self, cmp: List[Bool], _true: Float32, _false: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)
- **_true** (`Float32`)
- **_false** (`Self`)

**Returns:**

`Self`

```mojo
fn where(self, cmp: List[Bool], _true: Self, _false: Self) -> Self
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
fn argwhere_l(self, cmp: List[Bool]) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **cmp** (`List`)

**Returns:**

`List`

### `C_transpose`

```mojo
fn C_transpose(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `F_transpose`

```mojo
fn F_transpose(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `T`

```mojo
fn T(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `asorder`

```mojo
fn asorder(self, order: String) -> Self
```

**Args:**

- **self** (`Self`)
- **order** (`String`)

**Returns:**

`Self`

### `cumsum`

```mojo
fn cumsum(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `sum`

```mojo
fn sum(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn sum(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `mean`

```mojo
fn mean(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn mean(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `mean_slow`

```mojo
fn mean_slow(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

### `mean_slow0`

```mojo
fn mean_slow0(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `std`

```mojo
fn std(self, correction: Bool = False) -> Float32
```

**Args:**

- **self** (`Self`)
- **correction** (`Bool`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn std(self, _mean: Float32, correction: Bool = False) -> Float32
```

**Args:**

- **self** (`Self`)
- **_mean** (`Float32`)
- **correction** (`Bool`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn std(self, axis: Int, correction: Bool = False) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)
- **correction** (`Bool`)

**Returns:**

`Self`

**Raises:**

```mojo
fn std(self, axis: Int, _mean: Self, correction: Bool = False) -> Self
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
fn std_slow(self, _mean: Float32) -> Float32
```

**Args:**

- **self** (`Self`)
- **_mean** (`Float32`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn std_slow(self, axis: Int, _mean: Self) -> Self
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
fn abs(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `log`

```mojo
fn log(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `sqrt`

```mojo
fn sqrt(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `exp`

```mojo
fn exp(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

### `argmin`

```mojo
fn argmin(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

```mojo
fn argmin(self, axis: Int) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`List`

### `argmax`

```mojo
fn argmax(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

```mojo
fn argmax(self, axis: Int) -> List[Int]
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`List`

### `argmax_f`

```mojo
fn argmax_f(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

### `argsort`

```mojo
fn argsort[ascending: Bool = True](self) -> List[Scalar[DType.index]]
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
fn argsort_inplace[ascending: Bool = True](mut self) -> List[Scalar[DType.index]]
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
fn min(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn min(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `max`

```mojo
fn max(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

```mojo
fn max(self, axis: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **axis** (`Int`)

**Returns:**

`Self`

**Raises:**

### `reshape`

```mojo
fn reshape(self, height: Int, width: Int) -> Self
```

**Args:**

- **self** (`Self`)
- **height** (`Int`)
- **width** (`Int`)

**Returns:**

`Self`

### `cov`

```mojo
fn cov(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `lu_factor`

```mojo
@staticmethod
fn lu_factor(mut A, piv: UnsafePointer[Int], N: Int)
```

**Args:**

- **A** (`Self`)
- **piv** (`UnsafePointer`)
- **N** (`Int`)

**Raises:**

### `lu_solve`

```mojo
@staticmethod
fn lu_solve(A, piv: UnsafePointer[Int], b: Self, mut x: Self, N: Int, Mi: Int)
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
fn solve(var A, b: Self) -> Self
```

**Args:**

- **A** (`Self`)
- **b** (`Self`)

**Returns:**

`Self`

**Raises:**

### `inv`

```mojo
fn inv(self) -> Self
```

**Args:**

- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `eye`

```mojo
@staticmethod
fn eye(n: Int, order: String = "c") -> Self
```

**Args:**

- **n** (`Int`)
- **order** (`String`)

**Returns:**

`Self`

### `norm`

```mojo
fn norm(self) -> Float32
```

**Args:**

- **self** (`Self`)

**Returns:**

`Float32`

**Raises:**

### `outer`

```mojo
fn outer(self, rhs: Self) -> Self
```

**Args:**

- **self** (`Self`)
- **rhs** (`Self`)

**Returns:**

`Self`

**Raises:**

### `concatenate`

```mojo
fn concatenate(self, rhs: Self, axis: Int) -> Self
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
fn bincount(self) -> List[Int]
```

**Args:**

- **self** (`Self`)

**Returns:**

`List`

**Raises:**

### `unique`

```mojo
fn unique(self) -> List[List[Int]]
```

**Args:**

- **self** (`Self`)

**Returns:**

`List`

### `is_uniquef`

```mojo
fn is_uniquef(self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

### `zeros`

```mojo
@staticmethod
fn zeros(height: Int, width: Int, order: String = "c") -> Self
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
fn ones(height: Int, width: Int, order: String = "c") -> Self
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
fn full(height: Int, width: Int, val: Float32, order: String = "c") -> Self
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
fn fill_zero(self)
```

**Args:**

- **self** (`Self`)

### `fill`

```mojo
fn fill(self, val: Float32)
```

**Args:**

- **self** (`Self`)
- **val** (`Float32`)

### `random`

```mojo
@staticmethod
fn random(height: Int, width: Int, order: String = "c") -> Self
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
fn rand_choice(arang: Int, size: Int, replace: Bool = True, seed: Bool = True) -> List[Scalar[DType.index]]
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
fn linspace(start: Float32, stop: Float32, num: Int, order: String = "c") -> Self
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
fn from_numpy(np_arr: PythonObject, order: String = "c") -> Self
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
fn to_numpy(self) -> PythonObject
```

Converts the matrix to a numpy array.

**Args:**

- **self** (`Self`)

**Returns:**

`PythonObject`: The numpy array.

**Raises:**

### `cast_ptr`

```mojo
fn cast_ptr[des: DType](self) -> UnsafePointer[Scalar[des]]
```

**Parameters:**

- **des** (`DType`)

**Args:**

- **self** (`Self`)

**Returns:**

`UnsafePointer`

### `write_to`

```mojo
fn write_to[W: Writer](self, mut writer: W)
```

**Parameters:**

- **W** (`Writer`)

**Args:**

- **self** (`Self`)
- **writer** (`W`)

### `__str__`

```mojo
fn __str__(self) -> String
```

**Args:**

- **self** (`Self`)

**Returns:**

`String`


