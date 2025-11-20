Mojo function

# `svm_decision_function`

```mojo
fn svm_decision_function(model: svm_model, x: UnsafePointer[svm_node, origin_of(MutOrigin.external)]) -> Tuple[UnsafePointer[Float64, origin_of(MutOrigin.external)], Int]
```

**Args:**

- **model** (`svm_model`)
- **x** (`UnsafePointer`)

**Returns:**

`Tuple`

