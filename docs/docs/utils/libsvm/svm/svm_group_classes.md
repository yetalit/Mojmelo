Mojo function

# `svm_group_classes`

```mojo
fn svm_group_classes(prob: svm_problem, mut nr_class_ret: Int, mut label_ret: UnsafePointer[Int, MutExternalOrigin], mut start_ret: UnsafePointer[Int, MutExternalOrigin], mut count_ret: UnsafePointer[Int, MutExternalOrigin], perm: UnsafePointer[Scalar[DType.index], MutExternalOrigin])
```

**Args:**

- **prob** (`svm_problem`)
- **nr_class_ret** (`Int`)
- **label_ret** (`UnsafePointer`)
- **start_ret** (`UnsafePointer`)
- **count_ret** (`UnsafePointer`)
- **perm** (`UnsafePointer`)

