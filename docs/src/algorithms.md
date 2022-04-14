# Algorithms

Here we describe each approach available in `InferOpt.jl`.

## Combinatorial problems as layers

### Linear formulation

As we stated in the [Mathematical background](@ref), our package is centered around the integration of LP layers such as

```math
\theta \longmapsto \hat{y}(\theta) = \arg\max_{y \in \mathcal{Y}} \theta^T y \tag{LP}
```

into machine learning pipelines. Here, ``\theta`` is a cost vector (obtained as the output of the encoder ``\varphi_w``), while ``\mathcal{Y}`` is a finite subset of ``\mathbb{R}^{d}``.
Since the optimum of an LP is always reached at a vertex of the feasible polytope, we can start by replacing ``\mathcal{Y}`` with its convex hull ``\mathcal{C} = \mathrm{conv}(\mathcal{Y})``.

Note that the set of feasible solutions ``\mathcal{Y}`` and its convex hull ``\mathcal{C}`` may depend on the instance ``x``.
In that case, we use the notations ``\mathcal{Y}(x)`` and ``\mathcal{C}(x)``, but the exposition doesn't change.

### Implementation doesn't matter

Importantly, our framework does not constrain the actual procedure used to find a solution ``\hat{y}(\theta)``.
As long as the problem we solve corresponds to the maximization of a linear function over a convex polytope ``\mathcal{C}``, anything is fair game, and we do not care about the implementation details.

- Example 1: If we consider a Mixed Integer Linear Program (MILP), the convex hull of the integer solutions often cannot be expressed in a concise way. In that case, we will most likely use a MILP solver on ``\mathcal{Y}`` instead of an LP solver on ``\mathcal{C}``. Still, the problem can be described as a maximization over the continuous polytope ``\mathcal{C}``.
  
- Example 2: In some applications, we don't even have to rely on mathematical programming solvers such as CPLEX or Gurobi. For instance, Dijkstra's algorithm for shortest paths or the Edmonds-Karp algorithm for maximum flows can also be used to tackle LPs with specific structure.

### The problem with differentiability

Let us suppose that the problem (LP) is one of many layers in a (deep) learning pipeline.
To lean the parameters of the other layers, we would like to use a gradient algorithm, which requires the whole pipeline to be differentiable.
Unfortunately, the argmin of an LP is a piecewise-constant function, able to jump discontinuously between polytope vertices with very small shifts in the cost vector ``\theta``.

Therefore, a major contribution of our package consists in a toolbox for constructing differentiable approximations of discrete optimization layers.

We now present the catalogue of methods available in `InferOpt.jl`, along with the differentiation formulae.
In what follows, the `pullback` function computes vector-jacobian products (see [Defining chain rules](@ref) to understand why that is a central notion in differentiable programming).

## Differentiating through an argmax

### Piecewise-linear interpolation

A first option is to construct a piecewise-linear interpolation, whose distance from the piecewise-constant argmax is controlled by a smoothing parameter ``\lambda``.
Here is the formula for the backward pass:
```math
\texttt{pullback}(\delta_y) = \frac{1}{\lambda}(\hat{y}(\theta + \lambda \delta_y) - \hat{y}(\theta))
```

> [Differentiation of Blackbox Combinatorial Solvers](https://arxiv.org/abs/1912.02175)

### Regularized prediction

Another solution is to regularize the predictor ``\hat{y}(\theta) = \arg\max_{y \in \mathcal{C}} \theta^T y`` using a regularization function ``\Omega`` on the output space.
This is expressed as follows:

```math
\hat{y}_{\Omega}(\theta) = \arg\max_{y \in \mathcal{C}} \theta^T y - \Omega(y)
```

> [Learning with Fenchel-Young Losses](https://arxiv.org/abs/1901.02324)

A special case of this approach is regularization by stochastic perturbation.
Let ``\varepsilon > 0`` and ``Z`` be a random vector with negative log-density ``\nu(z)``: we can define the perturbed optimizer

```math
\hat{y}_{\varepsilon}(\theta) = \mathbb{E}_Z \big[ \arg\max_{y \in \mathcal{C}} (\theta + \varepsilon Z)^T y \big]
```

In this setting, the function ``\Omega`` has no explicit expression
However, we have a formula for the Jacobian of ``\hat{y}_{\varepsilon}(\theta)`` with respect to ``\theta``:

```math
J_\theta \hat{y}_{\varepsilon}(\theta) = \mathbb{E}_Z \big[ \hat{y}(\theta + \varepsilon Z) \nabla \nu(Z)^T / \varepsilon \big]
```

Therefore, we can compute a stochastic pullback function using samples ``Z_1,...,Z_M`` (which must the same as in the forward pass):

```math
\texttt{pullback}(\delta_y) = \frac{1}{\varepsilon M} \sum_{i=1}^{M} \big[\delta_y^T \hat{y}(\theta + \varepsilon Z_i)\big] \nabla \nu(Z_i)^T
```

> [Learning with Differentiable Perturbed Optimizers](https://arxiv.org/abs/2002.08676)

### Implicit differentiation

In this paragraph, the optimizer ``\hat{y}(\theta) \in \arg \max_{y \in \mathcal{C}} f(y, \theta)`` is not necessarily an LP.
However, we assume that every optimal solution must satisfy the following abstract conditions:

```math
F(\hat{y}(\theta), \theta) = 0
```

The implicit function theorem then gives the following relation between the jacobian matrices of ``\hat{y}``, ``F(\cdot, \theta)`` and ``F(y, \cdot)``:

```math
\underbrace{- J_y F(\hat{y}(\theta), \theta)}_{A} \cdot J_\theta \hat{y}(\theta) = \underbrace{J_\theta F(\hat{y}(\theta), \theta)}_{B}
```

The matrices ``A`` and ``B`` can be computed with automatic differentiation, but as it turns out, we don't need to store them entirely to compute vector-jacobian products.

> [Efficient and Modular Implicit Differentiation](http://arxiv.org/abs/2105.15183)

!!! note "Stay tuned!"
    This will soon be implemented thanks to the recent package [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl).

## Evaluating our predictions

Another crucial ingredient is a loss function that takes the structure of the problem into account.
Since any loss ``\ell(\theta)`` is a real-valued function, as soon as we have a subgradient ``g \in \partial \ell(\theta)``, we can automatically define a pullback as follows:

```math
\texttt{pullback}(\delta_\ell) = \delta_\ell g
```

### Smart "Predict, then Optimize"

Even when we know the true cost vector ``\bar{\theta}``, we may not simply want to minimize the error ``\lVert \theta - \bar{\theta} \rVert``.
What we are actually interested in is the impact of this error on the downstream optimization problem.
This is accurately measured by the SPO loss, of which the SPO+ loss is a convex surrogate:

```math
\ell^{SPO+}(\theta, \bar{\theta}) = (2\theta - \bar{\theta})^T\hat{y}(2 \theta - \bar{\theta}) + (\bar{\theta} - 2\theta)^T \hat{y}(\bar{\theta})
```

A subgradient with respect to ``\theta`` is given by

```math
2\hat{y}(2 \theta - \bar{\theta}) - 2\hat{y}(\bar{\theta}) \in \partial_{\theta} \ell^{SPO+}(\theta, \bar{\theta})
```

> [Smart "Predict, then Optimize"](https://arxiv.org/abs/1710.08005)

### Structured SVM

Suppose we define a "distance" function ``\Delta(\bar{y},y)`` on the output space.
The associated Structured SVM loss is given by

```math
\ell^{SSVM}(\theta, \bar{y}) = \max_{y \in \mathcal{C}} \Delta(\bar{y},y) + \theta^T(y - \bar{y})
```

> [Structured learning and prediction in computer vision](https://pub.ist.ac.at/~chl/papers/nowozin-fnt2011.pdf), Chapter 6

### Fenchel-Young losses

As soon as we have a regularized predictor, Fenchel-Young losses provide a systematic way to evaluate prediction quality in structured settings:

```math
\ell^{FY}(\theta, \bar{y})
= \Omega^*(\theta) + \Omega(\bar{y}) - \theta^T \bar{y}
= \max_{y \in \mathcal{C}} \left( \theta^T y - \Omega(y) \right) - \left( \theta^T \bar{y} - \Omega(\bar{y}) \right)
```

A subgradient with respect to ``\theta`` is given by the residual:

```math
    \hat{y}_{\Omega}(\theta) - \bar{y} \in \partial_{\theta} \ell^{FY}(\theta, \bar{y})
```

> [Learning with Fenchel-Young Losses](https://arxiv.org/abs/1901.02324)

In the case of stochastic perturbation, we cannot compute the full Fenchel-Young loss since ``\Omega(\bar{y})`` has no explicit formula. However, we can still optimize it without that term since it does not depend on ``\theta``:

```math
\ell^{FYP}(\theta, \bar{y}) = \mathbb{E}\big[\max_{y\in\mathcal{C}} (\theta + \varepsilon Z)^T y \big] - \theta^T \bar{y}
```

The subgradient expression thus remains unchanged.

> [Learning with Differentiable Perturbed Optimizers](https://arxiv.org/abs/2002.08676)