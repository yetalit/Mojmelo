Mojo function

# `svm_cross_validation`

```mojo
fn svm_cross_validation[k_t: Int](prob: svm_problem, param: svm_parameter, var nr_fold: Int, target: Optional[UnsafePointer[Float64, MutExternalOrigin]])
```

**Parameters:**

- **k_t** (`Int`)

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **nr_fold** (`Int`)
- **target** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)

