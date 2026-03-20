Mojo function

# `get_clusters`

```mojo
def get_clusters(tree: Dict[String, List[Scalar[DType.int]]], mut lambda_array: List[Float32], mut stability: Dict[Scalar[DType.int], Float32], cluster_selection_method: String = "eom", allow_single_cluster: Bool = False, match_reference_implementation: Bool = False, cluster_selection_epsilon: Float32 = 0, var max_cluster_size: Scalar[DType.int] = 0, cluster_selection_epsilon_max: Float32 = inf[DType.float32]()) -> Tuple[List[Scalar[DType.int]], List[Float32], List[Float32]]
```

**Args:**

- **tree** (`Dict`)
- **lambda_array** (`List`)
- **stability** (`Dict`)
- **cluster_selection_method** (`String`)
- **allow_single_cluster** (`Bool`)
- **match_reference_implementation** (`Bool`)
- **cluster_selection_epsilon** (`Float32`)
- **max_cluster_size** (`Scalar`)
- **cluster_selection_epsilon_max** (`Float32`)

**Returns:**

`Tuple`

**Raises:**

