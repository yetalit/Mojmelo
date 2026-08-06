Mojo function

# `trace_arg`

```mojo
fn def trace_arg(name: String, shape: IndexList[element_type=shape.element_type]) -> String
```

Helper to stringify the type and shape of a kernel argument for tracing.

**Args:**

- **name** (`String`): The name of the argument.
- **shape** (`IndexList[element_type=shape.element_type]`): The shape of the argument.

**Returns:**

`String`: A string representation of the argument with its shape.

```mojo
fn def trace_arg(name: String, shape: IndexList[element_type=shape.element_type], dtype: DType) -> String
```

Helper to stringify the type and shape of a kernel argument for tracing.

**Args:**

- **name** (`String`): The name of the argument.
- **shape** (`IndexList[element_type=shape.element_type]`): The shape of the argument.
- **dtype** (`DType`): The data type of the argument.

**Returns:**

`String`: A string representation of the argument with its shape and data type.

