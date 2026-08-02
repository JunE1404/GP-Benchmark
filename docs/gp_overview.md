# GP Variants & Optimizers in This Project

## GP regression in 5 lines

A GP prior over functions `f ~ GP(m(x), k(x, x'))` with Gaussian noise `y = f + ε` gives closed-form posterior:

```
mean(x*) = m(x*) + K*X (K + σ²I)⁻¹ (y − m)
var(x*)  = K** − K*X (K + σ²I)⁻¹ KX*
```

Training = maximizing the marginal likelihood
`log p(y) = −½ yᵀ K̃⁻¹ y − ½ log|K̃| − n/2·log 2π`, where `K̃ = K + σ²I`.

The vanilla cost is **O(n³)** time / **O(n²)** memory (forming and Cholesky-factorizing the `n×n` matrix) — that single number is what all four variants are fighting.

## The four GP variants

| Variant | Idea | Cost | Exact? |
|---|---|---|---|
| **ExactGP** | Full Cholesky of `K̃` | O(n³) | exact |
| **ExactGP-CG** | Same objective, but solves + logdet *approximated* | O(n² · #CG-iters) | approx (tolerance) |
| **SVGP** | Sparse variational: `m` inducing points | O(n·m²) | approx (variational) |
| **CAGP** | Budget computation explicitly via projections ("actions") | O(n² · P) via KeOps + O(P³) | approx (projection) |

### ExactGP (`exactgp.py`)
The textbook method: build `K̃`, Cholesky it once (`max_cholesky_size=inf`, `fast_computations` off), compute MLL and gradients exactly and deterministically. The optimization landscape is smooth — which is why LBFGS works well here. Practical ceiling ~10⁴ points on GPU.

### ExactGP-CG (`exactgp_conjg_gradients.py`)
Keeps the *exact* GP objective but replaces both expensive steps with approximations:
- **solves** (`K̃⁻¹y`) via conjugate gradients — only matrix–vector products, never the matrix
- **logdet** via *stochastic Lanczos quadrature* — random probe vectors, tridiagonalized

The catch: the logdet term uses **fresh random probes on every forward pass**, so the loss is *stochastic* — gradients are noisy. `cg_tolerance` sets how accurately CG solves (looser = faster + noisier). Because the objective is noisy, **Adam pairs with this variant far better than LBFGS** (the line search/curvature machinery gets garbage — exactly the NaN episode we hit).

### SVGP (`svgp.py`)
Variational inference: place `m` inducing points `u`, approximate the posterior with a Gaussian `q(u)`, and maximize the **ELBO**:
`ELBO = E_q[log p(y | f)] − KL(q(u) ‖ p(u))` — minus the full n×n matrix entirely. Cost drops to O(n·m²) (minibatched: O(m³) per batch). `kmeans.py` initializes inducing locations from k-means centroids instead of random picks — locations are learned anyway.

### CAGP (`cagp.py`, gpytorch `computation-aware-gps-v3` branch)
From Wenger et al. (2025): decide *up front* how much computation you'll spend, then make the best GP you can within that budget. A **projection/action matrix `S`** (block-sparse, `P` rows of "actions") compresses the training data; inference uses only the projected Gramian `SᵀK̃S` (P×P Cholesky, O(P³)). The **computation-aware ELBO** = expected log-likelihood under the projected data − KL to the prior. Kernel matrices are computed lazily via **KeOps** (`RBFKeops`, `matern2.5Keops`) — evaluated on the fly on GPU without ever materializing `n×n`. `approximation_size` = `P` = projection dimension.

### KeOps (all `*Keops` kernels)
Expression templates for kernels: `k(xi, xj)` is compiled and evaluated tile-by-tile on the GPU — no full matrix in memory. This is what makes 10⁴–10⁵ points feasible for the exact-family variants.

## The two optimizers

Both here optimize the *kernel hyperparameters* (lengthscale, outputscale, noise) — a tiny (≈4) parameter space, which is the key to everything below.

### Adam
Per-parameter adaptive learning rates from running averages of the gradient and its square (with bias correction). Tolerates **noisy or stochastic** gradients well (CG-MLL, minibatch SVGP). Steady but doesn't "converge fast"; lr=0.1 is a sane start. Papers: Kingma & Ba (2015).

### LBFGS
Quasi-Newton: builds an *approximation of the inverse Hessian* from the recent gradient history, so it converges in very few outer iterations on smooth problems — a 4-parameter MLL needs only ~1–3 outer steps. Two torch-specific gotchas this project hit:

1. **Default `line_search_fn=None` means NO line search**: step size is literally `t = lr`. `lr=0` → frozen (the sweep's "LR: 0" runs never trained); default `lr=1` can explode (the NaN Cholesky).
2. **`line_search_fn="strong_wolfe"`** does proper Wolfe-condition line search — then `lr` is only an initial guess and LBFGS behaves like the textbook method. This is the "trick" that only exists with the line search enabled.

Watch-out for both optimizers: **unconstrained MLL training has a degenerate optimum** — lengthscale → ∞, noise → its 1e-4 floor, kernel ≈ constant, loss flat. The optimizer happily converges there (the flat-loss runs). Always check the *final hyperparameters*, not just the loss curve.

## Resources (papers first)

**Background / exact GP**
- Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006) — free at gaussianprocess.org/gpml
- Quiñonero-Candela & Rasmussen, *A Unifying View of Sparse Approximate Gaussian Process Regression*, JMLR 2005

**CG / linear-operator machinery**
- Gardner et al., *GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration*, NeurIPS 2018 — arXiv:1811.06515 (CG, Lanczos, all the settings used here)
- Wang et al., *Exact Gaussian Processes on a Million Data Points*, NeurIPS 2019 — arXiv:1908.06752

**SVGP**
- Titsias, *Variational Learning of Inducing Variables in Sparse Gaussian Processes*, AISTATS 2009 — proceedings.mlr.press/v5/titsias09a.html
- Hensman et al., *Gaussian Processes for Big Data*, UAI 2013 — arXiv:1309.6835
- Hensman et al., *Scalable Variational Gaussian Process Classification*, AISTATS 2015 — arXiv:1411.2005

**CAGP**
- Wenger et al., *Computation-Aware Gaussian Processes: Model Selection Against Linear Models* — arXiv:2410.07785
- Wenger et al., *Posterior and Computational Uncertainty in Gaussian Processes*, NeurIPS 2022 — arXiv:2206.09884

**KeOps**
- Charlier et al., *Kernel Operations on the GPU, with Autodiff, without Memory Overruns*, JMLR 2021 — arXiv:2001.02419

**Optimizers**
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015 — arXiv:1412.6980
- Liu & Nocedal, *On the Limited Memory BFGS Method for Large Scale Optimization*, Math. Programming 1989
- Nocedal & Wright, *Numerical Optimization*, Springer (L-BFGS + line search theory)

**Practical**: docs.gpytorch.ai (exact GP / SVGP tutorials, `settings` reference — the CG tolerance bits are documented there).
