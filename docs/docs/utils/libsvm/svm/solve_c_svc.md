Mojo function

# `solve_c_svc`

```mojo
fn solve_c_svc[k_t: Int](prob: svm_problem, param: svm_parameter, alpha: Optional[UnsafePointer[Float64, MutExternalOrigin]], mut si: SolutionInfo, Cp: Float64, Cn: Float64)
```

**Parameters:**

- **k_t** (`Int`)

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **alpha** (`Optional[UnsafePointer[Float64, MutExternalOrigin]]`)
- **si** (`SolutionInfo`)
- **Cp** (`Float64`)
- **Cn** (`Float64`)

