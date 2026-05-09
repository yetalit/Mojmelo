Mojo function

# `svm_group_classes`

```mojo
fn svm_group_classes(prob: svm_problem, mut nr_class_ret: Int, mut label_ret: Optional[UnsafePointer[Int, MutExternalOrigin]], mut start_ret: Optional[UnsafePointer[Int, MutExternalOrigin]], mut count_ret: Optional[UnsafePointer[Int, MutExternalOrigin]], perm: Optional[UnsafePointer[Int, MutExternalOrigin]])
```

**Args:**

- **prob** (`svm_problem`)
- **nr_class_ret** (`Int`)
- **label_ret** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)
- **start_ret** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)
- **count_ret** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)
- **perm** (`Optional[UnsafePointer[Int, MutExternalOrigin]]`)

