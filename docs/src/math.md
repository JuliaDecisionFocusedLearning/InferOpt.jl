# Mathematical background

Our goal is make machine learning models more expressive by incomporating combinatorial optimization algorithms as layers.
For a broader perspective on the interactions between machine learning and combinatorial optimization, please refer to the following review papers:

!!! note "Reference"
    [Machine Learning for Combinatorial Optimization: A Methodological Tour d’Horizon](https://arxiv.org/abs/1811.06128)

!!! note "Reference"
    [End-to-end Constrained Optimization Learning: A Survey](https://arxiv.org/abs/2103.16378)


## Combinatorial optimization layers

### Linear formulation

Our package is centered around the integration of Linear Programs (LPs) such as
```math
\theta \longmapsto \hat{y}(\theta) = \arg\max_{y \in \mathcal{Y}} \theta^T y
```
into machine learning pipelines.
Here, ``\theta`` is an objective vector, while ``\mathcal{Y}`` is a finite subset of ``\mathbb{R}^{d}``.
Since the optimum of an LP is always reached at a vertex of the feasible polytope, we can start by replacing ``\mathcal{Y}`` with its convex hull ``\mathcal{C} = \mathrm{conv}(\mathcal{Y})``.

### Implementation doesn't matter

Our framework does not constrain the actual procedure used to find a solution ``\hat{y}(\theta)``.
As long as the problem we solve involves a linear objective over a polytope ``\mathcal{C}``, any optimization oracle is fair game, and we do not care about implementation details.

- Example 1: If we consider a Mixed Integer Linear Program (MILP), the convex hull of the integer solutions often cannot be expressed in a concise way. In that case, we will most likely use a MILP solver on ``\mathcal{Y}`` instead of an LP solver on ``\mathcal{C}``, even though both formulations are theoretically equivalent.
  
- Example 2: For some applications, we don't even have to rely on mathematical programming solvers such as CPLEX or Gurobi. For instance, Dijkstra's algorithm for shortest paths or the Edmonds-Karp algorithm for maximum flows can also be used to tackle LPs with specific structure.

### The problem with differentiability

Let us suppose that our LP is one of several layers in a machine learning pipeline.
To learn the parameters of the other layers, we would like to use a gradient algorithm, which requires the whole pipeline to be differentiable.
Unfortunately, the argmin of an LP is a piecewise-constant function, able to jump discontinuously between polytope vertices due to very small shifts in the objective vector ``\theta``.

The first contribution of `InferOpt.jl` consists in several methods for computing approximate differentials of LP layers.

## Loss functions for structured learning

In order to train our pipeline, we also need a loss function.
Ideally, this loss should be aware of the combinatorial optimization layer.
Furthermore, we want to adapt it to the kind of data at our disposal: do we have access to precomputed solutions ("learning by imitation") or just to previous problem instances ("learning by experience")?

The second contribution of `InferOpt.jl` is a catalogue of structured loss functions, many of them with nice properties such as differentiability or even convexity.