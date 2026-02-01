Mojo function

# `solve_c_svc`

```mojo
fn solve_c_svc(prob: svm_problem, param: svm_parameter, alpha: UnsafePointer[Float64, MutExternalOrigin], mut si: SolutionInfo, Cp: Float64, Cn: Float64)
```

**Args:**

- **prob** (`svm_problem`)
- **param** (`svm_parameter`)
- **alpha** (`UnsafePointer`)
- **si** (`SolutionInfo`)
- **Cp** (`Float64`)
- **Cn** (`Float64`)

