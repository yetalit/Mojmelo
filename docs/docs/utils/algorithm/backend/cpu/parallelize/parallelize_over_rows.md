Mojo function

# `parallelize_over_rows`

```mojo
fn def parallelize_over_rows[func: def(Int, Int) capturing thin -> None](shape: IndexList[element_type=shape.element_type], axis: Int, grain_size: Int)
```

Parallelize func over non-axis dims of shape.

**Parameters:**

- **func** (`def(Int, Int) capturing thin -> None`): Function to call on range of rows.

**Args:**

- **shape** (`IndexList[element_type=shape.element_type]`): Shape to parallelize over.
- **axis** (`Int`): Rows are slices along the axis dimension of shape.
- **grain_size** (`Int`): The minimum number of elements to warrant using an additional thread.

```mojo
fn def parallelize_over_rows[FuncType: def(Int, Int) -> None](func: FuncType, shape: IndexList[element_type=shape.element_type], axis: Int, grain_size: Int)
```

Parallelize func over non-axis dims of shape.

**Parameters:**

- **FuncType** (`def(Int, Int) -> None`): The body function type.

**Args:**

- **func** (`FuncType`): The closure carrying the captured state of the body function.
- **shape** (`IndexList[element_type=shape.element_type]`): Shape to parallelize over.
- **axis** (`Int`): Rows are slices along the axis dimension of shape.
- **grain_size** (`Int`): The minimum number of elements to warrant using an additional thread.

