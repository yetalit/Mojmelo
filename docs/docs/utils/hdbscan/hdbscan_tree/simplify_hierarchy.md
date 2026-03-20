Mojo function

# `simplify_hierarchy`

```mojo
def simplify_hierarchy(mut condensed_tree: Dict[String, List[Scalar[DType.int]]], mut lambda_array: List[Float32], persistence_threshold: Float32) -> Tuple[Dict[String, List[Scalar[DType.int]]], List[Float32]]
```

Remove leaves with persistence below threshold.

**Args:**

- **condensed_tree** (`Dict`)
- **lambda_array** (`List`)
- **persistence_threshold** (`Float32`)

**Returns:**

`Tuple`

**Raises:**

