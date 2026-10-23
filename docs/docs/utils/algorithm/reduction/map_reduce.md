Mojo function

# `map_reduce`

```mojo
fn def map_reduce[InputGenFnType: def[dtype: DType, width: Int](Int) -> SIMD[dtype, width], ReduceVecToVecFnType: def[acc_type: DType, dtype: DType, width: SIMDLength](SIMD[acc_type, width], SIMD[dtype, width]) -> SIMD[acc_type, width], OutputFnType: def[dtype_: DType, width: SIMDLength, alignment: Int](Int, SIMD[dtype_, width]) -> None, //, simd_width: SIMDLength, dtype: DType, acc_type: DType, reduce_vec_to_scalar_fn: def[dtype: DType, width: SIMDLength](SIMD[dtype, width]) thin -> Scalar[dtype]](input_gen_fn: InputGenFnType, reduce_vec_to_vec_fn: ReduceVecToVecFnType, output_fn: OutputFnType, length: Int, init: Scalar[acc_type]) -> Scalar[acc_type]
```

Performs a vectorized map-reduce operation over a sequence.

**Parameters:**

- **InputGenFnType** (`def[dtype: DType, width: Int](Int) -> SIMD[dtype, width]`): The type of `input_gen_fn` (inferred).
- **ReduceVecToVecFnType** (`def[acc_type: DType, dtype: DType, width: SIMDLength](SIMD[acc_type, width], SIMD[dtype, width]) -> SIMD[acc_type, width]`): The type of `reduce_vec_to_vec_fn` (inferred).
- **OutputFnType** (`def[dtype_: DType, width: SIMDLength, alignment: Int](Int, SIMD[dtype_, width]) -> None`): The type of `output_fn` (inferred).
- **simd_width** (`SIMDLength`): The SIMD vector width to use.
- **dtype** (`DType`): The data type of the input elements.
- **acc_type** (`DType`): The data type of the accumulator.
- **reduce_vec_to_scalar_fn** (`def[dtype: DType, width: SIMDLength](SIMD[dtype, width]) thin -> Scalar[dtype]`): Function that reduces a final vector to a scalar.

**Args:**

- **input_gen_fn** (`InputGenFnType`): Function that generates input values at each index.
- **reduce_vec_to_vec_fn** (`ReduceVecToVecFnType`): Function that reduces a vector into the accumulator.
- **output_fn** (`OutputFnType`): Function to output intermediate results.
- **length** (`Int`): The number of elements to process.
- **init** (`Scalar[acc_type]`): The initial accumulator value.

**Returns:**

`Scalar[acc_type]`: The final reduced scalar value.

