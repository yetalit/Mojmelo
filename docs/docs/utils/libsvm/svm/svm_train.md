Mojo function

# `svm_train`

```mojo
fn svm_train[k_t: Int](prob: svm_problem, param: svm_parameter) -> Optional[UnsafePointer[svm_model, MutExternalOrigin]]
```

**Parameters:**

- **k_t** (`Int`)

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)

**Returns:**

`Optional[UnsafePointer[svm_model, MutExternalOrigin]]`

