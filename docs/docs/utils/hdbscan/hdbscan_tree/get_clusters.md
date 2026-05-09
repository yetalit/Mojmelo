Mojo function

# `get_clusters`

```mojo
fn get_clusters(tree: Dict[String, List[Int]], mut lambda_array: List[Float32], mut stability: Dict[Int, Float32], cluster_selection_method: String = "eom", allow_single_cluster: Bool = False, match_reference_implementation: Bool = False, cluster_selection_epsilon: Float32 = 0, var max_cluster_size: Int = Int(0), cluster_selection_epsilon_max: Float32 = inf[DType.float32]()) -> Tuple[List[Int], List[Float32], List[Float32]]
```

**Args:**

- **tree** (`Dict[String, List[Int]]`)
- **lambda_array** (`List[Float32]`)
- **stability** (`Dict[Int, Float32]`)
- **cluster_selection_method** (`String`)
- **allow_single_cluster** (`Bool`)
- **match_reference_implementation** (`Bool`)
- **cluster_selection_epsilon** (`Float32`)
- **max_cluster_size** (`Int`)
- **cluster_selection_epsilon_max** (`Float32`)

**Returns:**

`Tuple[List[Int], List[Float32], List[Float32]]`

**Raises:**

