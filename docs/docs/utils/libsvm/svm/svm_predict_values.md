Mojo function

# `svm_predict_values`

```mojo
fn def svm_predict_values(model: svm_model, x: UnsafePointer[svm_node, MutUntrackedOrigin], dec_values: Optional[UnsafePointer[Float64, MutAnyOrigin]]) -> Float64
```

**Args:**

- **model** (`svm_model`)
- **x** (`UnsafePointer[svm_node, MutUntrackedOrigin]`)
- **dec_values** (`Optional[UnsafePointer[Float64, MutAnyOrigin]]`)

**Returns:**

`Float64`

