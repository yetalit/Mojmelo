Mojo function

# `svm_group_classes`

```mojo
fn def svm_group_classes(prob: svm_problem, mut nr_class_ret: Int, mut label_ret: Optional[UnsafePointer[Int, MutUntrackedOrigin]], mut start_ret: Optional[UnsafePointer[Int, MutUntrackedOrigin]], mut count_ret: Optional[UnsafePointer[Int, MutUntrackedOrigin]], perm: UnsafePointer[Int, MutUntrackedOrigin])
```

**Args:**

- **prob** (`svm_problem`)
- **nr_class_ret** (`Int`)
- **label_ret** (`Optional[UnsafePointer[Int, MutUntrackedOrigin]]`)
- **start_ret** (`Optional[UnsafePointer[Int, MutUntrackedOrigin]]`)
- **count_ret** (`Optional[UnsafePointer[Int, MutUntrackedOrigin]]`)
- **perm** (`UnsafePointer[Int, MutUntrackedOrigin]`)

