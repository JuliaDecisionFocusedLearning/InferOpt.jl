# Introduction

The goal of InferOpt.jl is to provide tools to use discrete functions into machine learning pipelines.

### How the math works

Consider the following combinatorial optimization oracle:
```math
    f\colon \theta \longmapsto \arg \max_{y \in \mathcal{Y}} g(y, \theta)
```
where $\mathcal{Y} \subset \mathbb{R}^d$ is a finite set of feasible solutions, $\theta$ is a vector input parameter, and $g$ is a scalar function.

!!! example
    Note that many discrete functions can be formulated this way. For instance:
    - The regular argmax function.
    - Ranking or sorting a vector.
    - Optimization algorithms over graphs, such as shortest paths algorithms.
    - Linear program (LP) or mixed integer linear program (MILP).

Unfortunately, the optimal solution $f(\theta)$ is often a piecewise constant function of $\theta$, which means its derivative is either zero or undefined.
Starting from a given oracle for $f$, InferOpt.jl approximates it with a differentiable "layer", whose derivatives convey meaningful slope information.
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

### How the code works

Since we want our package to be as generic as possible, we don't make any assumptions on the oracle used for $f$.
That way, the best solver can be selected for each use case.
We only ask the user to provide a black box function called `maximizer`, taking $\theta$ as argument and returning $f(\theta)$.

This function is then wrapped into a callable Julia `struct`, which can be used (for instance) within neural networks from the [Flux.jl](https://github.com/FluxML/Flux.jl) or [Lux.jl](https://github.com/LuxDL/Lux.jl) library.
To achieve this compatibility, we leverage Julia's automatic differentiation (AD) ecosystem, which revolves around the [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) package.

### Using InferOpt in machine learning applications

InferOpt can be used to make machine learning pipelines more expressive by incorporating combinatorial optimization layers.

Typically, a Combinatorial Optimization algorithm can be put as the last layer of a Machine Learning pipeline, after a statistical model (e.g. a neural network).
This gives discrete structured outputs, which enables several applications such as:
- Using an argmax layer instead of a softmax
- Learning to rank/learning to rank
- Multilabel classification
- Pathfinding on a map from an image

### Using InferOpt in combinatorial optimization applications

InferOpt has been mostly used to help solve hard variants of well-known combinatorial optimization problems.

For instance, it can be used for:
- Stochastic vehicle scheduling
- Multi-stage dynamic vehicle routing
- Two-stage minimum weight spanning tree
- Single machine scheduling
