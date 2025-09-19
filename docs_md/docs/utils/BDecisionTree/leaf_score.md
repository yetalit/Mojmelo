Mojo function

# `leaf_score`

```mojo
fn leaf_score(reg_lambda: SIMD[float32, 1], reg_alpha: SIMD[float32, 1], g: Matrix, h: Matrix) -> SIMD[float32, 1]
```

Given the gradient and hessian of a tree leaf, return the prediction (score) at this leaf. The score is -G/(H+λ).

**Args:**

- **reg_lambda** (`SIMD`)
- **reg_alpha** (`SIMD`)
- **g** (`Matrix`)
- **h** (`Matrix`)

**Returns:**

`SIMD`

**Raises:**

