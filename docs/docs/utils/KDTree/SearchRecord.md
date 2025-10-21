Mojo struct

# `SearchRecord`

```mojo
@memory_only
struct SearchRecord
```

## Aliases

- `__del__is_trivial = True`

## Fields

- **qv** (`UnsafePointer[Float32]`)
- **dim** (`Int`)
- **rearrange** (`Bool`)
- **nn** (`UInt`)
- **ballsize** (`Float32`)
- **centeridx** (`Int`)
- **correltime** (`Int`)
- **result** (`UnsafePointer[KDTreeResultVector]`)
- **data** (`UnsafePointer[Matrix]`)
- **ind** (`UnsafePointer[List[Scalar[DType.index]]]`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, qv_in: NDBuffer[DType.float32, 1, origin], tree_in: KDTree[sort_results, rearrange], result_in: KDTreeResultVector)
```

**Args:**

- **qv_in** (`NDBuffer`)
- **tree_in** (`KDTree`)
- **result_in** (`KDTreeResultVector`)
- **self** (`Self`)

**Returns:**

`Self`


