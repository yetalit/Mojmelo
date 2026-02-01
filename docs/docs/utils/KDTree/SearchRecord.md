Mojo struct

# `SearchRecord`

```mojo
@memory_only
struct SearchRecord
```

## Aliases

- `__del__is_trivial = True`

## Fields

- **qv** (`UnsafePointer[Float32, MutAnyOrigin]`)
- **dim** (`Int`)
- **rearrange** (`Bool`)
- **nn** (`UInt`)
- **ballsize** (`Float32`)
- **centeridx** (`Int`)
- **correltime** (`Int`)
- **result** (`UnsafePointer[KDTreeResultVector, MutAnyOrigin]`)
- **data** (`UnsafePointer[Matrix, MutAnyOrigin]`)
- **ind** (`UnsafePointer[List[Scalar[DType.index]], MutAnyOrigin]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, qv_in: Span[Float32, MutAnyOrigin], mut tree_in: KDTree[sort_results, rearrange], mut result_in: KDTreeResultVector)
```

**Args:**

- **qv_in** (`Span`)
- **tree_in** (`KDTree`)
- **result_in** (`KDTreeResultVector`)
- **self** (`Self`)

**Returns:**

`Self`


