# API Reference

## Index

```@index
Modules = [InferOpt]
```

## Probability distributions

```@autodocs
Modules = [InferOpt]
Pages = ["utils/probability_distribution.jl", "utils/composition.jl"]
```

## Interpolation

!!! note "Reference"
    [Differentiation of Blackbox Combinatorial Solvers](https://arxiv.org/abs/1912.02175)

```@autodocs
Modules = [InferOpt]
Pages = ["interpolation/interpolation.jl"]
```

## Smart "Predict, then Optimize"

!!! note "Reference"
    [Smart "Predict, then Optimize"](https://arxiv.org/abs/1710.08005)

```@autodocs
Modules = [InferOpt]
Pages = ["spo/spoplus_loss.jl"]
```

## Structured Support Vector Machines

!!! note "Reference"
    [Structured learning and prediction in computer vision](https://pub.ist.ac.at/~chl/papers/nowozin-fnt2011.pdf), Chapter 6

```@autodocs
Modules = [InferOpt]
Pages = ["ssvm/isbaseloss.jl", "ssvm/ssvm_loss.jl", "ssvm/zeroone_baseloss.jl"]
```

## Frank-Wolfe

!!! note "Reference"
    [Efficient and Modular Implicit Differentiation](http://arxiv.org/abs/2105.15183)

```@autodocs
Modules = [InferOpt]
Pages = ["frank_wolfe/frank_wolfe_utils.jl", "frank_wolfe/differentiable_frank_wolfe.jl"]
```

## Regularized optimizers

!!! note "Reference"
    [Learning with Fenchel-Young Losses](https://arxiv.org/abs/1901.02324)

```@autodocs
Modules = [InferOpt]
Pages = ["regularized/frank_wolfe.jl", "regularized/isregularized.jl", "regularized/soft_argmax.jl", "regularized/sparse_argmax.jl", "regularized/regularized_generic.jl", "regularized/regularized_utils.jl"]
```

## Perturbed optimizers

!!! note "Reference"
    [Learning with Differentiable Perturbed Optimizers](https://arxiv.org/abs/2002.08676)

```@autodocs
Modules = [InferOpt]
Pages = ["perturbed/abstract_perturbed.jl", "perturbed/additive.jl", "perturbed/composition.jl", "perturbed/multiplicative.jl"]
```

## Fenchel-Young losses

!!! note "Reference"
    [Learning with Fenchel-Young Losses](https://arxiv.org/abs/1901.02324)

```@autodocs
Modules = [InferOpt]
Pages = ["fenchel_young/fenchel_young.jl"]
```
