# InferOpt.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/InferOpt.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDecisionFocusedLearning.github.io/InferOpt.jl/dev)
[![Build Status](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaDecisionFocusedLearning/InferOpt.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaDecisionFocusedLearning/InferOpt.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Overview

InferOpt.jl is a toolbox for using combinatorial optimization algorithms within machine learning pipelines.

It allows you to create differentiable layers from optimization oracles that do not have meaningful derivatives.
Typical examples include mixed integer linear programs or graph algorithms.

## Getting started

To install the stable version, open a Julia REPL and run the following command:

```julia
julia> using Pkg; Pkg.add("InferOpt")
```

To install the development version, run this command instead:

```julia
julia> using Pkg; Pkg.add(url="https://github.com/JuliaDecisionFocusedLearning/InferOpt.jl")
```

## Citing us

If you use our package in your research, please cite the following paper:

> [Learning with Combinatorial Optimization Layers: a Probabilistic Approach](https://arxiv.org/abs/2207.13513) - Guillaume Dalle, Léo Baty, Louis Bouvier and Axel Parmentier (2022)

## Related packages

The following libraries implement similar functionalities:

- [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl): automatic differentiation of implicit functions 
- [DiffOpt.jl](https://github.com/jump-dev/DiffOpt.jl): differentiating convex optimization programs w.r.t. program parameters
- [JAXopt](https://github.com/google/jaxopt): hardware accelerated, batchable and differentiable optimizers in JAX