Mojo module

# `linalg_core`

## Aliases

- `RealScalar = Float64`
- `REAL_EPSILON = 2.2204460492503131E-16`
- `REAL_MIN = 2.2250738585072014E-308`
- `SQRT_REAL_MAX = sqrt(SIMD(1.7976931348623157E+308))`
- `INFO_SUCCESS = ComputationInfo(Int(0))`
- `INFO_NUMERICAL_ISSUE = ComputationInfo(Int(1))`
- `INFO_NO_CONVERGENCE = ComputationInfo(Int(2))`
- `INFO_INVALID_INPUT = ComputationInfo(Int(3))`
- `SIMD_WIDTH = (Int(4) * simd_width_of[DType.float64]()) if CompilationTarget.is_apple_silicon() else (Int(2) * simd_width_of[DType.float64]())`

## Structs

- [`ComputationInfo`](ComputationInfo.md)
- [`Vec`](Vec.md)
- [`IVec`](IVec.md)
- [`Mat`](Mat.md)

## Functions

- [`swap_vecs`](swap_vecs.md)
- [`vec_dot`](vec_dot.md)
- [`reverse_cols`](reverse_cols.md)
- [`mat_transpose`](mat_transpose.md)
- [`mat_scale`](mat_scale.md)
- [`mat_identity`](mat_identity.md)
- [`embed_topleft`](embed_topleft.md)
- [`matmul`](matmul.md)

