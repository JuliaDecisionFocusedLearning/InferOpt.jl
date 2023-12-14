# # Basics

using InferOpt
using LinearAlgebra: I, dot
# using Flux
using Zygote

# ## Differentiating through argmax

#=
```math
\arg\max_y \theta^\top y
```
=#

function onehot_argmax(θ)
    n = length(θ)
    #simplex = [I(n)[i, :] for i in 1:n]
    return I(n)[argmax(θ), :]# simplex[argmax(θ)]
end

#
n = 4
θ = randn(n)
#
onehot_argmax(θ)
#
Zygote.jacobian(onehot_argmax, θ)[1]

#
# softmax(θ)
soft_argmax(θ)
#
sparse_argmax(θ)

# Zygote.jacobian(softmax, θ)[1]
Zygote.jacobian(soft_argmax, θ)[1]
#
Zygote.jacobian(sparse_argmax, θ)[1]

#
perturbed = PerturbedAdditive(onehot_argmax; nb_samples=100, seed=0)
#
perturbed(θ)
#
Zygote.jacobian(perturbed, θ)[1]
