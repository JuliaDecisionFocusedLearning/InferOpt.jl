# Mathematical background

Here we describe the theoretical framework in which `InferOpt.jl` operates.
Our goal is make machine learning models more expressive by incorporating combinatorial optimization algorithms as layers.

For a broader perspective on the interactions between machine learning and combinatorial optimization, please refer to the following review papers:

> [Machine Learning for Combinatorial Optimization: A Methodological Tour d’Horizon](https://arxiv.org/abs/1811.06128)

> [End-to-end Constrained Optimization Learning: A Survey](https://arxiv.org/abs/2103.16378)

## General setting

Let ``\mathcal{X}`` be a set of instances, in a very broad sense.
For each instance ``x \in \mathcal{X}``, we want to predict an output ``z \in \mathcal{Z}(x)`` which minimizes a given cost function ``c(z)``.
The cost may also depend on ``x``, but we omit this dependency for notational simplicity:

```math
\min_{z \in \mathcal{Z}(x)} c(z)
```

In many real-world applications, the set of feasible predictions is combinatorially large: well-known examples include rankings, paths, flows, etc.
To tackle such scenarios, a common approach in the literature is to delegate all combinatorial aspects to a surrogate optimization problem, which is typically a Linear Program (LP).

Let us therefore introduce a combinatorially large but _structured_ set ``\mathcal{Y}(x)``.
By _structured_, we mean that there is a natural embedding of ``\mathcal{Y}(x)`` in ``\mathbb{R}^{d(x)}``, and that we have efficient algorithms to solve the following problem: [^1]

[^1]: We switched to maximization here to ensure consistency with the literature on structured learning.

```math
\max_{y \in \mathcal{Y}(x)} \theta^T y \tag{LP}
```

## Structured learning pipeline

The main purpose of `InferOpt.jl` is to integrate the optimization problem (LP) into a learning pipeline such as this one:

```math
\xrightarrow[\text{Instance}]{x \in \mathcal{X}}
\fbox{Encoder $\varphi_w$}
\xrightarrow[\text{Cost vector}]{\theta \in \mathbb{R}^{d(x)}}
\fbox{Optimizer}
\xrightarrow[\text{Solution}]{y \in \mathcal{Y}(x)}
\fbox{Decoder $\psi$}
\xrightarrow[\text{Output}]{z \in \mathcal{Z}(x)}
```

What is going on here?

1. The _encoder_ ``\varphi_w`` transforms an instance ``x`` in ``\mathcal{X}`` into a cost vector ``\theta = \varphi_w(x)`` in ``\mathbb{R}^{d(x)}``. It can be any machine learning algorithm parameterized by a set of weights ``w``, like a GLM or neural network.

2. The _optimizer_ solves (LP) and returns an optimal solution ``\hat{y}(\theta) \in \arg\max_{y \in \mathcal{Y}(x)} \theta^T y``.

3. The _decoder_ ``\psi`` turns the solution ``\hat{y}(\theta)`` into an output ``\hat{z} \in \mathcal{Z}(x)``. It is usually a handcrafted repair or local search heuristic with no learnable parameters.

Our goal is to learn the encoder weights ``w`` based on previous instances, so that we may solve future instances by plugging them into our pipeline.

In many of the methods we describe, the decoder either doesn't exist or doesn't play a central role, which is why we ignore it in what follows by simply taking ``\mathcal{Y}(x) = \mathcal{Z}(x)``.

## Learning by experience

Let us denote by ``f_w = \hat{y} \circ \varphi_w`` our prediction pipeline.
A natural instinct would be to find the weights ``w`` that minimize the _empirical regret_, and prevent overfitting with a weight regularization term ``g(w)``:

```math
R_n(w) = \frac{1}{n} \sum_{i=1}^n c(f_w(x_i)) + g(w)
```

Directly minimizing ``R_n(w)`` can be referred to as _learning by experience_, since the algorithm is only guided by the knowledge of past instances ``x_i``.

Unfortunately, the function ``R_n`` depends on ``c`` (which can be an expensive black box) and ``f_w`` (which is, in general, piecewise-constant).
As a result, it is not easy to handle directly, and we often need additional guidance.

## Learning by imitation

For each instance ``x_i \in \mathcal{X}``, our training set may also contain some _target_ ``\xi_i \in \Xi(x_i)`` that orients us towards desirable behavior.
Learning by imitation is cheaper since expensive black box computations can happen offline, outside of the training loop.

There are two main cases:

1. When ``\xi_i = \bar{y}_i`` is a precomputed optimal solution (as in [Structured SVM](@ref) and [Fenchel-Young losses](@ref)).
2. When ``\xi_i = \bar{\theta}_i`` is the true cost vector, from which we can recover ``\bar{y}_i \in \arg\max_{y \in \mathcal{Y}(x)} \langle \bar{\theta}_i, y \rangle`` (as in [Smart "Predict, then Optimize"](@ref)).

The cost function is usually adapted to accept the target as an additional argument, and possibly convexified.