Mojo struct

# `HDBSCANBoruvka`

```mojo
@memory_only
struct HDBSCANBoruvka
```

## Fields

- **tree** (`UnsafePointer[KDTreeBoruvka, MutAnyOrigin]`)
- **n** (`Int`)
- **dim** (`Int`)
- **min_samples** (`Int`)
- **alpha** (`Float32`)
- **num_components** (`Int`)
- **candidate_point** (`List[Int]`)
- **candidate_neighbor** (`List[Int]`)
- **candidate_dist** (`List[Float32]`)
- **component_bound** (`List[Float32]`)
- **u_f** (`UnionFind`)
- **u_f_finds** (`List[Int]`)
- **edges** (`Matrix`)
- **num_edges** (`Int`)
- **component_of_point** (`List[Int]`)
- **component_of_node** (`List[Int]`)
- **component_remap** (`List[Int]`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, t: UnsafePointer[KDTreeBoruvka, MutAnyOrigin], min_samples: Int = 5, alpha: Float32 = 1)
```

**Args:**

- **t** (`UnsafePointer[KDTreeBoruvka, MutAnyOrigin]`)
- **min_samples** (`Int`)
- **alpha** (`Float32`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `mr_rdist`

```mojo
fn mr_rdist(self, var d2: Float32, p: Int, q: Int) -> Float32
```

**Args:**

- **self** (`Self`)
- **d2** (`Float32`)
- **p** (`Int`)
- **q** (`Int`)

**Returns:**

`Float32`

### `update_components_and_nodes`

```mojo
fn update_components_and_nodes(mut self)
```

**Args:**

- **self** (`Self`)

**Raises:**

### `boruvka_query`

```mojo
fn boruvka_query(mut self)
```

**Args:**

- **self** (`Self`)

**Raises:**

### `merge_components`

```mojo
fn merge_components(mut self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

**Raises:**

### `spanning_tree`

```mojo
fn spanning_tree(mut self) -> Matrix
```

**Args:**

- **self** (`Self`)

**Returns:**

`Matrix`

**Raises:**


