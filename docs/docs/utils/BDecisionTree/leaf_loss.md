Mojo function

# `leaf_loss`

```mojo
fn leaf_loss(reg_lambda: SIMD[float32, 1], reg_alpha: SIMD[float32, 1], g: Matrix, h: Matrix) -> SIMD[float32, 1]
```

Given the gradient and hessian of a tree leaf, return the minimized loss at this leaf. The minimized loss is -0.5*G^2/(H+λ). .

**Args:**

- **reg_lambda** (`SIMD`)
- **reg_alpha** (`SIMD`)
- **g** (`Matrix`)
- **h** (`Matrix`)

**Returns:**

`SIMD`

**Raises:**

