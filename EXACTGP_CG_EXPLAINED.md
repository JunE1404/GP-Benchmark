# Exact Gaussian Processes with Conjugate Gradients

## Table of Contents

1. [What is a Gaussian Process?](#1-what-is-a-gaussian-process)
2. [The Training Objective: Marginal Log-Likelihood](#2-the-training-objective-marginal-log-likelihood)
3. [The Computational Bottleneck](#3-the-computational-bottleneck)
4. [Conjugate Gradients to the Rescue](#4-conjugate-gradients-to-the-rescue)
5. [The Log-Determinant Problem](#5-the-log-determinant-problem)
6. [Putting It All Together: The Loss Function](#6-putting-it-all-together-the-loss-function)
7. [Prediction After Training](#7-prediction-after-training)
8. [Autograd and Why Gradient Flow Matters](#8-autograd-and-why-gradient-flow-matters)
9. [Walkthrough of the Code](#9-walkthrough-of-the-code)

---

## 1. What is a Gaussian Process?

A Gaussian Process (GP) is a way of thinking about functions probabilistically. Instead of saying "the function is `f(x) = ax + b`", a GP says: *"at every point `x`, the function value `f(x)` is a random variable, and any finite collection of these variables has a joint Gaussian distribution."*

A GP is fully defined by two things:

- **A mean function** `m(x)` — our prior belief about `f(x)` before seeing any data.
- **A covariance (kernel) function** `k(x, x')` — how strongly correlated `f(x)` and `f(x')` are. Points that are close in input space should have similar function values.

Given training inputs `X = [x_1, ..., x_N]` and noisy observations `y = [y_1, ..., y_N]`, a GP assumes:

```
y_i = f(x_i) + ε_i,    ε_i ~ N(0, σ²)
```

where `σ²` (the **likelihood noise**) accounts for measurement noise. The function values at the training points follow:

```
f(X) ~ N(m(X), K)
```

where `K` is the `N × N` kernel matrix with `K_ij = k(x_i, x_j)`.

---

## 2. The Training Objective: Marginal Log-Likelihood

During training, we want to tune the **hyperparameters** — the kernel parameters (lengthscales, output scale) and the noise variance `σ²` — so that the model best explains the observed data.

The probability of seeing the observed targets `y` given the inputs `X` and hyperparameters `θ` is:

```
p(y | X, θ) = N(y | m(X), K + σ²I)
```

It's Gaussian because:
- `f(X)` is Gaussian with covariance `K`.
- The noise `ε` is Gaussian with covariance `σ²I`.
- The sum of two independent Gaussians is Gaussian.

Taking the log and expanding gives the **negative marginal log-likelihood** (the loss we minimize):

```
L = -log p(y | X, θ)
  = ½ (y - m(X))ᵀ (K + σ²I)⁻¹ (y - m(X))
  + ½ log |K + σ²I|
  + ½ N log(2π)
```

### What each term means

| Term | Name | What it does |
|---|---|---|
| `(y - m)ᵀ (K + σ²I)⁻¹ (y - m)` | **Data fit** | Penalizes functions that don't pass through the data. Scales inversely with the covariance — high-noise regions get less penalty. |
| `log \|K + σ²I\|` | **Complexity penalty** | Penalizes overly complex kernels. A kernel with large lengthscales produces a smooth, low-rank matrix with a small determinant; a wiggly kernel has a large determinant. |
| `N log(2π)` | **Normalization constant** | Irrelevant for optimization — just a constant shift. |

Minimizing `L` means finding hyperparameters that balance **fitting the data** against **model complexity**.

---

## 3. The Computational Bottleneck

To evaluate `L`, we need two things:

1. **`α = (K + σ²I)⁻¹ (y - m)`** — solve a linear system.
2. **`log |K + σ²I|`** — compute the log-determinant.

The standard way to do both is the **Cholesky decomposition**:

1. Factor `K + σ²I = L Lᵀ` where `L` is lower triangular.
2. Solve `L Lᵀ α = (y - m)` by forward/backward substitution (O(N²)).
3. `log |K + σ²I| = 2 Σ log(L_ii)` (trivial once the Cholesky is done).

**Problem**: Cholesky is O(N³). With `N = 10,000` points, that's a trillion operations per iteration — infeasible.

**This is where Conjugate Gradients comes in.**

---

## 4. Conjugate Gradients to the Rescue

### 4.1 What CG does

Conjugate Gradients is an iterative algorithm for solving linear systems of the form:

```
A x = b
```

where `A` is symmetric and positive-definite. It does **not** need `A` explicitly — it only needs to compute **matrix-vector products** `A @ v`.

For us, `A = K + σ²I` and `b = y - m(X)`.

### 4.2 How CG works (intuition)

CG starts with `x = 0` and refines it iteratively:

1. Compute the residual `r = b - A x` (how far we are from the true solution).
2. Pick a search direction `p` that is **conjugate** (A-orthogonal) to all previous directions.
3. Take the optimal step along `p` to minimize the error in the A-norm.
4. Repeat until the residual is small enough.

After `k` iterations, CG gives the exact solution for a system with only `k` distinct eigenvalues. In practice, it converges in far fewer than `N` iterations, especially when eigenvalues cluster.

Each iteration does:
- One matrix-vector product: `A @ p` (O(N²) for a dense matrix).
- A few vector dot-products and axpy operations (O(N)).

So `k` iterations cost O(k · N²) instead of O(N³) for Cholesky.

### 4.3 The algorithm (simplified)

```
Given A, b, max_iter, tol:

  x = 0
  r = b - A @ x
  p = r
  rsold = r · r

  for i in 1..max_iter:
    Ap = A @ p
    α = rsold / (p · Ap)
    x = x + α p
    r = r - α Ap
    rsnew = r · r

    if sqrt(rsnew) < tol: stop

    p = r + (rsnew / rsold) p
    rsold = rsnew

  return x
```

Every operation shown here is a standard PyTorch operation — no `out=` arguments, so gradients flow through correctly.

### 4.4 Why CG instead of torch.linalg.solve?

| Method | Time | Memory | Best for |
|---|---|---|---|
| `torch.linalg.solve` (Cholesky/LU) | O(N³) | O(N²) | N < ~5000 |
| CG (k iterations) | O(k · N²) | O(N²) | N > ~5000, well-conditioned systems |
| Structured CG (e.g. with KeOps) | O(k · N · d) | O(N · d) | Very large N with structured kernels |

CG avoids the O(N³) factorization. For N = 10,000 and k = 50, it's ~200× cheaper than a direct solve.

---

## 5. The Log-Determinant Problem

Even with CG for the linear solve, we still need `log |K + σ²I|`. The standard Cholesky approach would be O(N³).

In this implementation, we use a pragmatic shortcut: **`torch.linalg.slogdet`** on the dense matrix. This is:

- **Exact** (not an approximation).
- **Autograd-compatible** (gradients flow through it).
- **O(N³)** — so it remains the bottleneck.

For larger N, you'd replace this with a stochastic log-det estimator (like Hutchinson's trace estimator + Lanczos), which is O(k · N²) — matching the CG cost.

### Why we can use a dense matrix

For moderate N (up to ~10,000), storing the N×N kernel matrix in GPU memory takes ~800 MB in float32. That's feasible on most modern GPUs. The two bottlenecks are:

- **Solving** the linear system: CG makes this O(k · N²).
- **Computing the log-det**: `slogdet` is still O(N³), but it's a single call (one Cholesky per loss evaluation).

For truly large N (» 10,000), you'd need both CG and a stochastic log-det estimator to avoid the O(N³) cost of `slogdet`.

---

## 6. Putting It All Together: The Loss Function

```python
def _compute_loss_cg(x_train, y_train):
    N = x_train.shape[0]
    noise = self.likelihood.noise

    K = self.covar_module(x_train, x_train).to_dense()
    K_noisy = K + noise * I + jitter * I

    y_centered = y_train - self.mean_module(x_train)

    # 1) Solve (K + σ²I) α = (y - m) using CG
    alpha = _cg(lambda v: K_noisy @ v, y_centered, max_iter=50)

    # 2) Compute log-determinant
    logdet = torch.linalg.slogdet(K_noisy)[1]

    # 3) Assemble loss
    loss = 0.5 * (y_centered @ alpha + logdet + N * log(2π))

    return loss
```

### The 0.5 factor and the three terms

```
loss = 0.5 * [ y_centered @ alpha     +     logdet     +   N * log(2π) ]

               |                              |                 |
               |                              |                 └─ Normalization constant
               |                              |                    (ignored by optimizer)
               |                              └─ Complexity penalty
               |                                 (large → model is too complex)
               └─ Data-fit term
                  (large → predictions miss the data)
```

Both `y_centered @ alpha` and `logdet` depend on the hyperparameters through `K` and the mean function, so gradients flow backward to update them.

---

## 7. Prediction After Training

Once the hyperparameters are optimized, we need to make predictions at test points `X_*`.

### The posterior predictive distribution

```
f(X_*) | X, y, X_*  ~  N(μ_*, Σ_*)
```

where:

```
μ_* = m(X_*) + K(X_*, X) (K + σ²I)⁻¹ (y - m(X))
```

```
Σ_* = K(X_*, X_*) - K(X_*, X) (K + σ²I)⁻¹ K(X, X_*)
```

### What the code does

```python
def predict(self, x):
    # Cache α from the last training step
    prior_mean = self.mean_module(x)
    k_x_train = self.covar_module(x, self.train_data[0])

    pred_mean = prior_mean + k_x_train @ self._alpha
    pred_covar = self.covar_module(x)   # no uncertainty reduction!

    return MultivariateNormal(pred_mean, pred_covar)
```

**Important caveat**: The code uses `self.covar_module(x)` (the **prior** covariance at the test points) rather than the full posterior covariance `Σ_*`, which should include the `K(X_*, X) K⁻¹ K(X, X_*)` correction term. This means the predictive variance does **not** reflect the uncertainty reduction from training data — the predicted variance at training points is the same as at far-away points.

---

## 8. Autograd and Why Gradient Flow Matters

PyTorch builds a **computational graph** of every tensor operation. When you call `.backward()`, it walks backward through this graph to compute gradients.

The original crash happened because `linear_operator.utils.linear_cg` uses:

```python
torch.mul(a, b, out=storage)
```

The `out=` argument tells PyTorch to write the result into a pre-allocated buffer. **PyTorch's autograd does not track `out=` operations**, so if `a` or `b` requires gradients (which they do — they depend on kernel parameters), autograd cannot trace through them and raises an error.

The fix (`_cg` solver) uses only:

- `torch.dot` — dot product
- `torch.add` / `torch.sub` via `+` and `-`
- `torch.mul` via `*`

All of these are tracked by autograd, so gradients flow through the entire CG solve. This means the optimizer can differentiate through the solve to update kernel hyperparameters correctly.

Similarly, `torch.linalg.slogdet` is a first-class PyTorch operation with a registered backward pass, so it also supports autograd.

---

## 9. Walkthrough of the Code

### 9.1 `_cg(matmul, b, max_iter, tol)` — The custom CG solver

```
x = 0              ── Initial guess
r = b - A @ x      ── Initial residual
p = r               ── Initial search direction

Loop:
  Ap = A @ p        ── One matrix-vector multiply (the expensive step)
  α = r·r / p·Ap    ── Optimal step size
  x += α p          ── Update solution
  r -= α Ap         ── Update residual
  if |r| < tol: stop
  β = r·r / old_r·r
  p = r + β p       ── New conjugate direction
```

### 9.2 `ExactGPConjGradients.__init__` — Setting up

Stores training/test data, kernel, mean, and likelihood. Moves everything to GPU if requested. Calls the parent `gpytorch.models.ExactGP.__init__` which registers the training data for gpytorch's internal use.

### 9.3 `run_training(optimizer, iterations)` — The training loop

```
for i in range(iterations):
  optimizer.zero_grad()
  loss = _compute_loss_cg(train_X, train_y)
  loss.backward()
  optimizer.step()
```

At each iteration:

1. **Zero the gradients** from the previous step.
2. **Compute the loss** (marginal log-likelihood) using CG.
3. **Backpropagate**: compute gradients of the loss w.r.t. all hyperparameters (lengthscale, output scale, noise).
4. **Step the optimizer**: Adam (or LBFGS) updates the hyperparameters to reduce the loss.

### 9.4 `predict(x)` — Making predictions

Uses the cached `self._alpha = (K + σ²I)⁻¹ (y - m)` from the final training step to compute `μ_*` for new test points.

### 9.5 The computational cost per iteration

| Step | Cost | Notes |
|---|---|---|
| Build K | O(N² · d) | Evaluate kernel for all pairs (d = input dim) |
| CG solve (50 iters) | O(50 · N²) | 50 matrix-vector products |
| logdet | O(N³) | `slogdet` does a Cholesky internally |
| Backward pass | O(N³) | Through the logdet + through CG ops |

The Cholesky inside `slogdet` is the bottleneck. For very large N, replacing it with a stochastic estimator (e.g. Lanczos + Hutchinson trace) would move the cost to O(k · N²), matching CG.

---

## Summary

| Concept | What it means for the GP |
|---|---|
| **Gaussian Process** | A distribution over functions; any finite set of points is jointly Gaussian. |
| **Kernel / covariance** | Encodes smoothness assumptions via pairwise similarity of inputs. |
| **Marginal log-likelihood** | The training loss: balances data fit vs. model complexity. |
| **Conjugate Gradients** | Solves `K α = (y - m)` in O(k · N²) without computing `K⁻¹` explicitly. |
| **Log-determinant** | Complexity penalty; computed via `slogdet` (exact, O(N³)). |
| **Autograd** | PyTorch's automatic differentiation; our custom CG uses only graph-tracked ops. |
| **Prediction** | `μ_* = m(X_*) + K(X_*, X) α` — a linear combination of training observations. |
