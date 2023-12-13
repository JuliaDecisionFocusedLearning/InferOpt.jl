# API Reference

```@docs
InferOpt
```

## Generic optimization layers
```@docs
PerturbedAdditive
PerturbedMultiplicative
PerturbedOracle
RegularizedFrankWolfe
```

## Problem specific optimization layers
```@docs
SoftArgmax
SparseArgmax
SoftRank
SoftSort
IdentityRelaxation
Interpolation
```

## Losses
```@docs
FenchelYoungLoss
SPOPlusLoss
StructuredSVMLoss
ImitationLoss
Pushforward
```

## Generalized maximizer
```@docs
GeneralizedMaximizer
```

## Public abstract interfaces
```@docs
AbstractRegularized
AbstractRegularizedGeneralizedMaximizer
```

## Function versions of specific layers
```@docs
soft_argmax
sparse_argmax
soft_rank
soft_sort
InferOpt.ZeroOneImitationLoss
InferOpt.ZeroOneStructuredSVMLoss
```

## Internals

### Types

```@autodocs
Modules = [InferOpt]
Public = false
Order = [:type]
```

### Miscellaneous
```@docs
FixedAtomsProbabilityDistribution
```

### Functions
```@autodocs
Modules = [InferOpt]
Public = false
Order = [:function]
```

### Is exported but should be private?

```@docs
soft_rank_kl
soft_rank_l2
soft_sort_kl
soft_sort_l2
half_square_norm
negative_shannon_entropy
compute_expectation
objective_value
one_hot_argmax
ranking
shannon_entropy
compute_probability_distribution(::InferOpt.AbstractPerturbed, ::AbstractArray)
compute_probability_distribution
compute_probability_distribution(::Pushforward, ::Any)
get_y_true(::NamedTuple)
get_y_true
```

## Index

```@index
Modules = [InferOpt]
```
