Mojo function

# `svm_decision_function`

```mojo
fn svm_decision_function(model: svm_model, x: Optional[UnsafePointer[svm_node, MutExternalOrigin]]) -> Tuple[Optional[UnsafePointer[Float64, MutExternalOrigin]], Int]
```

**Args:**

- **model** (`svm_model`)
- **x** (`Optional[UnsafePointer[svm_node, MutExternalOrigin]]`)

**Returns:**

`Tuple[Optional[UnsafePointer[Float64, MutExternalOrigin]], Int]`

