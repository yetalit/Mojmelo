Mojo struct

# `DecisionTree`

```mojo
@memory_only
struct DecisionTree[criterion: String = "gini"]
```

A decision tree supporting both classification and regression.

## Aliases

- `loss_func = gini if (criterion == String("gini")) else mse_loss if (criterion == String("mse")) else entropy`
- `c_precompute = gini_precompute if (criterion == String("gini")) else entropy_precompute`
- `r_precompute: __mlir_type.`!kgen.func.literal<:!lit.fn<[2]("size": !lit.struct<_std::_builtin::_simd::_SIMD<:!lit.struct<_std::_builtin::_dtype::_DType> {:dtype f32}, :!lit.struct<_std::_builtin::_int::_Int> {1}>>, "sum": !lit.struct<_std::_builtin::_simd::_SIMD<:!lit.struct<_std::_builtin::_dtype::_DType> {:dtype f32}, :!lit.struct<_std::_builtin::_int::_Int> {1}>>, "sum_sq": !lit.struct<_std::_builtin::_simd::_SIMD<:!lit.struct<_std::_builtin::_dtype::_DType> {:dtype f32}, :!lit.struct<_std::_builtin::_int::_Int> {1}>>, ?, "__error__": !lit.ref<!lit.struct<_std::_builtin::_error::_Error>, mut *[0,0]> byref_error, "__result__": !lit.ref<:meta<!lit.struct<_std::_builtin::_simd::_SIMD<:!lit.struct<_std::_builtin::_dtype::_DType> {:dtype f32}, :!lit.struct<_std::_builtin::_int::_Int> {1}>>> sugar_alias(*"Float32`0x21", _std::_builtin::_simd::_SIMD<:!lit.struct<_std::_builtin::_dtype::_DType> {:dtype f32}, :!lit.struct<_std::_builtin::_int::_Int> {1}>), mut *[0,1]> byref_result) throws -> i1> #kgen.func.symbol<_mojmelo::_utils::_utils::_"mse_loss_precompute(::SIMD[::DType(float32), ::Int(1)],::SIMD[::DType(float32), ::Int(1)],::SIMD[::DType(float32), ::Int(1)])">>` = fn_literal`
- `MODEL_ID = 9`

## Parameters

- **criterion** (`String`): The function to measure the quality of a split:
    For classification -> 'entropy', 'gini';
    For regression -> 'mse'.

## Fields

- **min_samples_split** (`Int`): The minimum number of samples required to split an internal node.
- **max_depth** (`Int`): The maximum depth of the tree.
- **n_feats** (`Int`): The number of features to consider when looking for the best split.
- **root** (`Optional[UnsafePointer[Node, MutAnyOrigin]]`)

## Implemented traits

`AnyType`, `CV`, `Copyable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Movable`

## Methods

### `__init__`

```mojo
fn __init__(out self, min_samples_split: Int = 2, max_depth: Int = 100, n_feats: Int = -1, random_state: Int = 42)
```

**Args:**

- **min_samples_split** (`Int`)
- **max_depth** (`Int`)
- **n_feats** (`Int`)
- **random_state** (`Int`)
- **self** (`Self`)

**Returns:**

`Self`

```mojo
fn __init__(out self, params: Dict[String, String])
```

**Args:**

- **params** (`Dict[String, String]`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `__del__`

```mojo
fn __del__(deinit self)
```

**Args:**

- **self** (`Self`)

### `fit`

```mojo
fn fit(mut self, X: Matrix, y: Matrix)
```

Build a decision tree from the training set.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y** (`Matrix`)

**Raises:**

### `fit_weighted`

```mojo
fn fit_weighted(mut self, X: Matrix, y_with_weights: Matrix)
```

Build a decision tree from a weighted training set.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)
- **y_with_weights** (`Matrix`)

**Raises:**

### `predict`

```mojo
fn predict(self, X: Matrix) -> Matrix
```

Predict class or regression value for X.

**Args:**

- **self** (`Self`)
- **X** (`Matrix`)

**Returns:**

`Matrix`: The predicted values.

**Raises:**

### `save`

```mojo
fn save(self, path: String)
```

Save model data necessary for prediction to the specified path.

**Args:**

- **self** (`Self`)
- **path** (`String`)

**Raises:**

### `load`

```mojo
@staticmethod
fn load(path: String) -> Self
```

Load a saved model from the specified path for prediction.

**Args:**

- **path** (`String`)

**Returns:**

`Self`

**Raises:**


