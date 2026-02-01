Mojo struct

# `HDBSCANBoruvka`

```mojo
@memory_only
struct HDBSCANBoruvka
```

## Aliases

- `__del__is_trivial = False`

## Fields

- **tree** (`UnsafePointer[KDTreeBoruvka, MutAnyOrigin]`)
- **n** (`Int`)
- **dim** (`Int`)
- **min_samples** (`Int`)
- **alpha** (`Float32`)
- **num_components** (`Int`)
- **component_of_point** (`List[Scalar[DType.index]]`)
- **component_of_node** (`List[Scalar[DType.index]]`)
- **component_remap** (`List[Scalar[DType.index]]`)
- **candidate_point** (`List[Scalar[DType.index]]`)
- **candidate_neighbor** (`List[Scalar[DType.index]]`)
- **candidate_dist** (`List[Float32]`)
- **u_f** (`UnionFind`)
- **u_f_finds** (`List[Int]`)
- **edges** (`Matrix`)
- **num_edges** (`Int`)

## Implemented traits

`AnyType`, `ImplicitlyDestructible`

## Methods

### `__init__`

```mojo
fn __init__(out self, t: UnsafePointer[KDTreeBoruvka, MutAnyOrigin], min_samples: Int = 5, alpha: Float32 = 1)
```

**Args:**

- **t** (`UnsafePointer`)
- **min_samples** (`Int`)
- **alpha** (`Float32`)
- **self** (`Self`)

**Returns:**

`Self`

**Raises:**

### `mr_rdist`

```mojo
fn mr_rdist(self, var d2: Float32, p: Scalar[DType.index], q: Scalar[DType.index]) -> Float32
```

**Args:**

- **self** (`Self`)
- **d2** (`Float32`)
- **p** (`Scalar`)
- **q** (`Scalar`)

**Returns:**

`Float32`

### `update_components_and_nodes`

```mojo
fn update_components_and_nodes(mut self) -> Int
```

**Args:**

- **self** (`Self`)

**Returns:**

`Int`

**Raises:**

### `dual_tree_traversal`

```mojo
fn dual_tree_traversal(mut self, node1: Int, node2: Int)
```

**Args:**

- **self** (`Self`)
- **node1** (`Int`)
- **node2** (`Int`)

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


