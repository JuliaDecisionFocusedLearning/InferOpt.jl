# InferOpt.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://axelparmentier.github.io/InferOpt.jl/dev)
[![Build Status](https://github.com/axelparmentier/InferOpt.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/axelparmentier/InferOpt.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/axelparmentier/InferOpt.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/axelparmentier/InferOpt.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

## Overview

`InferOpt.jl` is a toolbox for using combinatorial optimization algorithms within machine learning pipelines.

It allows you to differentiate through things that should not be differentiable, such as Mixed Integer Linear Programs or graph algorithms.

> This package is at a very early development stage, so proceed with caution!

## Getting started

To install the stable version, open a Julia REPL and run the following command:

```julia
julia> using Pkg; Pkg.add("InferOpt")
```

To install the development version, run this command instead:

```julia
julia> using Pkg; Pkg.add(url="https://github.com/axelparmentier/InferOpt.jl")
```
