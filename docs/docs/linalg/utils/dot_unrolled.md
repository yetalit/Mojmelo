Mojo function

# `dot_unrolled`

```mojo
fn def dot_unrolled[dtype: DType, width: Int](pa: Pointer[Scalar[dtype], MutUntrackedOrigin], pb: Pointer[Scalar[dtype], MutUntrackedOrigin], count: Int) -> Scalar[dtype]
```

Dot product with 4 independent SIMD accumulators, so the FMA latency chain of a single accumulator doesn't bound throughput. `vectorize` handles whatever is left after the unrolled main loop.

**Parameters:**

- **dtype** (`DType`)
- **width** (`Int`)

**Args:**

- **pa** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`)
- **pb** (`Pointer[Scalar[dtype], MutUntrackedOrigin]`)
- **count** (`Int`)

**Returns:**

`Scalar[dtype]`

