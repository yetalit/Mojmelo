Mojo function

# `svm_predict_probability`

```mojo
fn svm_predict_probability(model: svm_model, x: UnsafePointer[svm_node, origin_of(MutOrigin.external)], prob_estimates: UnsafePointer[Float64, origin_of(MutOrigin.external)]) -> Float64
```

**Args:**

- **model** (`svm_model`)
- **x** (`UnsafePointer`)
- **prob_estimates** (`UnsafePointer`)

**Returns:**

`Float64`

