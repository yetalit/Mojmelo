Mojo function

# `svm_predict_probability`

```mojo
fn svm_predict_probability(model: svm_model, x: UnsafePointer[svm_node, MutExternalOrigin], prob_estimates: UnsafePointer[Float64, MutExternalOrigin]) -> Float64
```

**Args:**

- **model** (`svm_model`)
- **x** (`UnsafePointer`)
- **prob_estimates** (`UnsafePointer`)

**Returns:**

`Float64`

