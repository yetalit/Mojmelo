Mojo function

# `solve_one_class`

```mojo
fn solve_one_class[k_t: Int](prob: svm_problem, param: svm_parameter, alpha: Optional[UnsafePointer[Float64, MutExternalOrigin]], mut si: SolutionInfo)
```

**Parameters:**

- **k_t** (`Int`)

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **alpha** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **si** (`SolutionInfo`)

