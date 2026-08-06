Mojo struct

# `SearchRecord`

```mojo
@memory_only
struct SearchRecord
```

## Fields

- **qv** (`Pointer[Float32, MutUntrackedOrigin]`)
- **dim** (`Int`)
- **rearrange** (`Bool`)
- **nn** (`UInt`)
- **ballsize** (`Float32`)
- **centeridx** (`Int`)
- **correltime** (`Int`)
- **result** (`Pointer[KDTreeResultVector, MutUntrackedOrigin]`)
- **data** (`Pointer[Matrix, MutUntrackedOrigin]`)
- **ind** (`Pointer[List[Int], MutUntrackedOrigin]`)

## Implemented traits

`AnyType`, `Deinitable`, `Movable`

## Methods

### `__init__`

```mojo
fn def __init__(out self, qv_in: Span[Float32, MutUntrackedOrigin], tree_in: KDTree, result_in: KDTreeResultVector)
```

**Args:**

- **qv_in** (`Span[Float32, MutUntrackedOrigin]`)
- **tree_in** (`KDTree`)
- **result_in** (`KDTreeResultVector`)
- **self** (`Self`)

**Returns:**

`Self`


