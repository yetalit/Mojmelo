Mojo function

# `simplify_hierarchy`

```mojo
fn simplify_hierarchy(mut condensed_tree: Dict[String, List[Int]], mut lambda_array: List[Float32], persistence_threshold: Float32) -> Tuple[Dict[String, List[Int]], List[Float32]]
```

Remove leaves with persistence below threshold.

**Args:**

- **condensed_tree** (`Dict[String, List[Int]]`)
- **lambda_array** (`List[Float32]`)
- **persistence_threshold** (`Float32`)

**Returns:**

`Tuple[Dict[String, List[Int]], List[Float32]]`

**Raises:**

