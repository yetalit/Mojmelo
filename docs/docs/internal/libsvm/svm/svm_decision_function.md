Mojo function

# `svm_decision_function`

```mojo
fn def svm_decision_function(model: svm_model, x: Pointer[svm_node, MutUntrackedOrigin]) -> Tuple[Pointer[Float64, MutUntrackedOrigin], Int]
```

**Args:**

- **model** (`svm_model`)
- **x** (`Pointer[svm_node, MutUntrackedOrigin]`)

**Returns:**

`Tuple[Pointer[Float64, MutUntrackedOrigin], Int]`

