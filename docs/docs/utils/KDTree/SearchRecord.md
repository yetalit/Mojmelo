Mojo struct

# `SearchRecord`

```mojo
@memory_only
struct SearchRecord
```

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
- **ind** (`UnsafePointer[List[Int], MutAnyOrigin]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, qv_in: Span[Float32, MutAnyOrigin], mut tree_in: KDTree, mut result_in: KDTreeResultVector)
```

**Args:**

- **qv_in** (`Span[Float32, MutAnyOrigin]`)
- **tree_in** (`KDTree`)
- **result_in** (`KDTreeResultVector`)
- **self** (`Self`)

**Returns:**

`Self`


