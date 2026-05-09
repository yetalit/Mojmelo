Mojo function

# `svm_predict_probability`

```mojo
fn svm_predict_probability(model: svm_model, x: Optional[UnsafePointer[svm_node, MutExternalOrigin]], prob_estimates: Optional[UnsafePointer[Float64, MutExternalOrigin]]) -> Float64
```

**Args:**

- **model** (`svm_model`)
- **x** (`Optional[UnsafePointer[svm_node, MutExternalOrigin]]`)
- **prob_estimates** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)

**Returns:**

`Float64`

