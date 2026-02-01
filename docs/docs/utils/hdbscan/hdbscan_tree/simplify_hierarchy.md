Mojo function

# `simplify_hierarchy`

```mojo
fn simplify_hierarchy(mut condensed_tree: Dict[String, List[Scalar[DType.index]]], mut lambda_array: List[Float32], persistence_threshold: Float32) -> Tuple[Dict[String, List[Scalar[DType.index]]], List[Float32]]
```

Remove leaves with persistence below threshold.

**Args:**

- **condensed_tree** (`Dict`)
- **lambda_array** (`List`)
- **persistence_threshold** (`Float32`)

**Returns:**

`Tuple`

**Raises:**

