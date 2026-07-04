Mojo function

# `svm_decision_function`

```mojo
fn def svm_decision_function(model: svm_model, x: UnsafePointer[svm_node, MutUntrackedOrigin]) -> Tuple[Optional[UnsafePointer[Float64, MutUntrackedOrigin]], Int]
```

**Args:**

- **model** (`svm_model`)
- **x** (`UnsafePointer[svm_node, MutUntrackedOrigin]`)

**Returns:**

`Tuple[Optional[UnsafePointer[Float64, MutUntrackedOrigin]], Int]`

