Mojo function

# `svm_group_classes`

```mojo
fn svm_group_classes(prob: svm_problem, mut nr_class_ret: Int, mut label_ret: UnsafePointer[Int, origin_of(MutOrigin.external)], mut start_ret: UnsafePointer[Int, origin_of(MutOrigin.external)], mut count_ret: UnsafePointer[Int, origin_of(MutOrigin.external)], perm: UnsafePointer[Scalar[DType.int], origin_of(MutOrigin.external)])
```

**Args:**

- **prob** (`svm_problem`)
- **nr_class_ret** (`Int`)
- **label_ret** (`UnsafePointer`)
- **start_ret** (`UnsafePointer`)
- **count_ret** (`UnsafePointer`)
- **perm** (`UnsafePointer`)

