Mojo function

# `svm_group_classes`

```mojo
fn def svm_group_classes(prob: svm_problem, mut nr_class_ret: Int, mut label_ret: Optional[Pointer[Int, MutUntrackedOrigin]], mut start_ret: Optional[Pointer[Int, MutUntrackedOrigin]], mut count_ret: Optional[Pointer[Int, MutUntrackedOrigin]], perm: Pointer[Int, MutUntrackedOrigin])
```

**Args:**

- **prob** (`svm_problem`)
- **nr_class_ret** (`Int`)
- **label_ret** (`Optional[Pointer[Int, MutUntrackedOrigin]]`)
- **start_ret** (`Optional[Pointer[Int, MutUntrackedOrigin]]`)
- **count_ret** (`Optional[Pointer[Int, MutUntrackedOrigin]]`)
- **perm** (`Pointer[Int, MutUntrackedOrigin]`)

