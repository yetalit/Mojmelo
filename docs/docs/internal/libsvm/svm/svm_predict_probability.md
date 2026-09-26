Mojo function

# `svm_predict_probability`

```mojo
fn def svm_predict_probability(model: svm_model, x: Pointer[svm_node, MutUntrackedOrigin], prob_estimates: Pointer[Float64, MutUntrackedOrigin]) -> Float64
```

**Args:**

- **model** (`svm_model`)
- **x** (`Pointer[svm_node, MutUntrackedOrigin]`)
- **prob_estimates** (`Pointer[Float64, MutUntrackedOrigin]`)

**Returns:**

`Float64`

