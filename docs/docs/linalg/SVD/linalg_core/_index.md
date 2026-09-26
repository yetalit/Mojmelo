Mojo module

# `linalg_core`

## Aliases

- `RealScalar = Float64`
- `REAL_EPSILON = 2.2204460492503131E-16`
- `REAL_MIN = 2.2250738585072014E-308`
- `SQRT_REAL_MAX = sqrt(SIMD(1.7976931348623157E+308))`
- `SIMD_WIDTH = (Int(4) * simd_width_of[DType.float64]()) if CompilationTarget.is_apple_silicon() else (Int(2) * simd_width_of[DType.float64]())`
- `PAR_ELEMS = 65536`

## Structs

- [`Vec`](Vec.md)
- [`IVec`](IVec.md)
- [`Mat`](Mat.md)

## Functions

- [`swap_vecs`](swap_vecs.md)
- [`vec_dot`](vec_dot.md)
- [`sub_inplace`](sub_inplace.md)
- [`reverse_cols`](reverse_cols.md)
- [`mat_transpose`](mat_transpose.md)
- [`mat_identity`](mat_identity.md)
- [`embed_topleft`](embed_topleft.md)
- [`matmul`](matmul.md)
- [`matmul_acc`](matmul_acc.md)

