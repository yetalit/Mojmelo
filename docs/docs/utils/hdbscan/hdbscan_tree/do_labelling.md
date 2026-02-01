Mojo function

# `do_labelling`

```mojo
fn do_labelling(tree: Dict[String, List[Scalar[DType.index]]], lambda_array: List[Float32], clusters: Set[Scalar[DType.index]], cluster_label_map: Dict[Scalar[DType.index], Int], allow_single_cluster: Int, cluster_selection_epsilon: Float32, match_reference_implementation: Int) -> List[Scalar[DType.index]]
```

**Args:**

- **tree** (`Dict`)
- **lambda_array** (`List`)
- **clusters** (`Set`)
- **cluster_label_map** (`Dict`)
- **allow_single_cluster** (`Int`)
- **cluster_selection_epsilon** (`Float32`)
- **match_reference_implementation** (`Int`)

**Returns:**

`List`

**Raises:**

