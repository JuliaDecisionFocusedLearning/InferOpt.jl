# API Reference

## Index

```@index
Modules = [InferOpt]
```

## Interpolation

!!! tip "Reference"
    [Differentiation of Blackbox Combinatorial Solvers](https://arxiv.org/abs/1912.02175)

```@autodocs
Modules = [InferOpt]
Pages = ["interpolation/interpolation.jl"]
```

## Smart "Predict, then Optimize"

!!! tip "Reference"
    [Smart "Predict, then Optimize"](https://arxiv.org/abs/1710.08005)

```@autodocs
Modules = [InferOpt]
Pages = ["spo/spoplus_loss.jl"]
```

## Structured Support Vector Machines

!!! tip "Reference"
    [Structured learning and prediction in computer vision](https://pub.ist.ac.at/~chl/papers/nowozin-fnt2011.pdf), Chapter 6

```@autodocs
Modules = [InferOpt]
Pages = ["ssvm/isbaseloss.jl", "ssvm/ssvm_loss.jl", "ssvm/zeroone_baseloss.jl"]
```

## Regularized optimizers

!!! tip "Reference"
    [Learning with Fenchel-Young Losses](https://arxiv.org/abs/1901.02324)

```@autodocs
Modules = [InferOpt]
Pages = ["regularized/frank_wolfe.jl", "regularized/isregularized.jl", "regularized/soft_argmax.jl", "regularized/sparse_argmax.jl", "regularized/regularized_utils.jl"]
```

## Perturbed optimizers

!!! tip "Reference"
    [Learning with Differentiable Perturbed Optimizers](https://arxiv.org/abs/2002.08676)

```@autodocs
Modules = [InferOpt]
Pages = ["perturbed/abstract_perturbed.jl", "perturbed/additive.jl", "perturbed/composition.jl", "perturbed/multiplicative.jl"]
```

## Fenchel-Young losses

!!! tip "Reference"
    [Learning with Fenchel-Young Losses](https://arxiv.org/abs/1901.02324)

```@autodocs
Modules = [InferOpt]
Pages = ["fenchel_young/fenchel_young.jl"]
```

## Implicit differentiation

!!! tip "Reference"
    [Efficient and Modular Implicit Differentiation](http://arxiv.org/abs/2105.15183)

!!! note "Stay tuned!"
    This will soon be implemented thanks to the recent package [ImplicitDifferentiation.jl](https://github.com/gdalle/ImplicitDifferentiation.jl).