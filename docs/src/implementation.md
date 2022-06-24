# Implementation

Here we describe the technical details of the `InferOpt.jl` codebase.

## Wrapping optimizers

In the [Mathematical background](@ref), we saw that our package provides a principled way to approximate combinatorial problems with machine learning.
More specifically, we implement several ways to convert combinatorial problems into differentiable layers of a machine learning pipeline.

Since we want our package to be as generic as possible, we don't make any assumptions on the kind of algorithm used to solve these combinatorial problems.
We only ask the user to provide a function called `maximizer`, which takes ``\theta`` as argument and returns a solution ``\hat{y}(\theta) = \arg\max_{y \in \mathcal{Y}} \theta^T y``.

This function is then wrapped into a callable Julia `struct` that can be used (for instance) within neural networks from the [`Flux.jl`](https://github.com/FluxML/Flux.jl) library.

!!! tip "Reference"
    [Flux: Elegant Machine Learning with Julia](https://joss.theoj.org/papers/10.21105/joss.00602)

## Leveraging Automatic Differentiation

To achieve this goal, we leverage Julia's Automatic Differentiation (AD) ecosystem, which revolves around the [`ChainRules.jl`](https://github.com/JuliaDiff/ChainRules.jl) package.
See the paper below for an overview of this ecosystem:

!!! tip "Reference"
    [AbstractDifferentiation.jl: Backend-Agnostic Differentiable Programming in Julia](http://arxiv.org/abs/2109.12449)

If you need a refresher on forward and reverse-mode AD, the following survey is a good starting point:

!!! tip "Reference"
    [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767)

In machine learning (especially deep learning), reverse-mode AD is by far the most common.
Therefore, as soon as we define a new type of layer, we must make it possible to compute the backward pass through this layer.
In other words, for each function ``b = f(a)``, we need to implement a "pullback function" that takes a sensitivity ``\delta_b`` and returns the associated sensitivity ``\delta_a = \delta_b \frac{\mathrm{d}b}{\mathrm{d}a}``.

See the [`ChainRules.jl` documentation](https://juliadiff.org/ChainRulesCore.jl/dev/index.html) for more details.
