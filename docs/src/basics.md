```@meta
EditURL = "../../examples/basics.jl"
```

# Basics

````@example basics
using InferOpt
using LinearAlgebra: I, dot
````

using Flux

````@example basics
using Zygote
````

## Differentiating through argmax

```math
\arg\max_y \theta^\top y
```

````@example basics
function onehot_argmax(θ)
    n = length(θ)
    #simplex = [I(n)[i, :] for i in 1:n]
    return I(n)[argmax(θ), :]# simplex[argmax(θ)]
end
````

````@example basics
n = 4
θ = randn(n)
````

````@example basics
onehot_argmax(θ)
````

````@example basics
Zygote.jacobian(onehot_argmax, θ)[1]
````

softmax(θ)

````@example basics
soft_argmax(θ)
````

````@example basics
sparse_argmax(θ)
````

Zygote.jacobian(softmax, θ)[1]

````@example basics
Zygote.jacobian(soft_argmax, θ)[1]
````

````@example basics
Zygote.jacobian(sparse_argmax, θ)[1]
````

````@example basics
perturbed = PerturbedAdditive(onehot_argmax; nb_samples=100, seed=0)
````

````@example basics
perturbed(θ)
````

````@example basics
Zygote.jacobian(perturbed, θ)[1]
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

