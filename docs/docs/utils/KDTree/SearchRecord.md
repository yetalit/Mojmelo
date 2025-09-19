Mojo struct

# `SearchRecord`

```mojo
@memory_only
struct SearchRecord
```

## Fields

- **qv** (`UnsafePointer[SIMD[float32, 1]]`)
- **dim** (`Int`)
- **rearrange** (`Bool`)
- **nn** (`UInt`)
- **ballsize** (`SIMD[float32, 1]`)
- **centeridx** (`Int`)
- **correltime** (`Int`)
- **result** (`UnsafePointer[KDTreeResultVector]`)
- **data** (`UnsafePointer[Matrix]`)
- **ind** (`UnsafePointer[List[SIMD[index, 1]]]`)

## Implemented traits

`AnyType`, `UnknownDestructibility`

## Methods

### `__init__`

```mojo
fn __init__(out self, qv_in: NDBuffer[float32, 1, origin], tree_in: KDTree[sort_results, rearrange], result_in: KDTreeResultVector)
```

**Args:**

- **qv_in** (`NDBuffer`)
- **tree_in** (`KDTree`)
- **result_in** (`KDTreeResultVector`)
- **self** (`Self`)

**Returns:**

`Self`


