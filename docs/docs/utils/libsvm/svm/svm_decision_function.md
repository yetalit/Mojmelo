Mojo function

# `svm_decision_function`

```mojo
fn svm_decision_function(model: svm_model, x: UnsafePointer[svm_node, MutExternalOrigin]) -> Tuple[UnsafePointer[Float64, MutExternalOrigin], Int]
```

**Args:**

- **model** (`svm_model`)
- **x** (`UnsafePointer`)

**Returns:**

`Tuple`

