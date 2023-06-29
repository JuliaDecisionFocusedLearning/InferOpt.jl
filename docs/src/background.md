# Background

The goal of InferOpt.jl is to make machine learning pipelines more expressive by incorporating combinatorial optimization layers.

## How the math works

Consider the following combinatorial optimization problem:
```math
    f\colon \theta \longmapsto \arg \max_{v \in \mathcal{V}} \theta^\top v
```
where $\mathcal{V} \subset \mathbb{R}^d$ is a finite set of feasible solutions, and $\theta$ is an objective vector.
Note that any linear program (LP) or mixed integer linear program (MILP) can be formulated this way.

Unfortunately, the optimal solution $f(\theta)$ is a piecewise constant function of $\theta$, which means its derivative is either zero or undefined.
Starting with an oracle for $f$, InferOpt.jl approximates it with a differentiable "layer", whose derivatives convey meaningful slope information.
Such a layer can then be used within a machine learning pipeline, and gradient descent will succeed.
InferOpt.jl also provides adequate loss functions for structured learning.

For more details on the theoretical aspects, you can check out our paper:

!!! note "Reference"
    [Learning with Combinatorial Optimization Layers: a Probabilistic Approach](https://arxiv.org/abs/2207.13513)

For a broader perspective on the interactions between machine learning and combinatorial optimization, please refer to the following surveys:

!!! note "Reference"
    [Machine Learning for Combinatorial Optimization: A Methodological Tour d’Horizon](https://arxiv.org/abs/1811.06128)

!!! note "Reference"
    [End-to-end Constrained Optimization Learning: A Survey](https://arxiv.org/abs/2103.16378)

## How the code works

Since we want our package to be as generic as possible, we don't make any assumptions on the oracle used for $f$.
That way, the best solver can be selected for each use case.
We only ask the user to provide a black box function called `maximizer`, taking $\theta$ as argument and returning $f(\theta)$.

This function is then wrapped into a callable Julia `struct`, which can be used (for instance) within neural networks from the [`Flux.jl`](https://github.com/FluxML/Flux.jl) library.
To achieve this compatibility, we leverage Julia's automatic differentiation (AD) ecosystem, which revolves around the [`ChainRules.jl`](https://github.com/JuliaDiff/ChainRules.jl) package.
See their [documentation](https://juliadiff.org/ChainRulesCore.jl/dev/index.html) for more details.