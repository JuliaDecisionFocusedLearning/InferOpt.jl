# # Basics: differentiating through argmax

#=
In this first tutorial, we show how to levarage InferOpt's features in order to compute
meaningfull derivatives of a very simple function: `argmax`.

Given a vector of float ``\theta``, the `argmax` method returns the index of the maximum value of ``\theta``:
=#
θ = [3.0, 3.1, 1.0]
argmax(θ)
# We can use the [`Zygote`](https://fluxml.ai/Zygote.jl/stable/) backward automatic differentiation library to try to differentiate through it
using Zygote
@show Zygote.gradient(argmax, θ)
# Zygote gradient is not defined 🙁. Let' try with [`ForwardDiff`](https://juliadiff.org/ForwardDiff.jl/stable/) instead:
using ForwardDiff
ForwardDiff.gradient(argmax, θ)
# Gradients are 0. That's expected, here is the argmax value when the second component changes:
using Plots
plot(x -> argmax([3.0, x, 1.0]), 1:0.01:5; xlabel="θ₂")
# Argmax is a discrete function, and is piecewise constant, therefore gradients are zero almost everywhere, which is not very useful in practice.
# This is where InferOpt can be used to smooth it and retrieve informative gradients.

# ## Differentiating through a black-box

# One tool that is very generic and can be used to differentiate through any discrte function
# are the [`InferOpt.AbstractPerturbed`](@ref) wrappers. The most common one is the [`PerturbedAdditive`](@ref),
# which can be used by wrapping the function you want to differentiate as follows:
using InferOpt: PerturbedAdditive
perturbed_argmax = PerturbedAdditive(argmax; ε=0.25, nb_samples=100000, seed=0)
# The resulting `perturbed_argmax` object is callable with the same arguments as the original function:
perturbed_argmax(θ)
# We can see that its output is continuous, unlike the original argmax function. The discontinuities are smoothed out:
plot(x -> argmax([3.0, x, 1.0]), 1:0.01:5; label="argmax", xlabel="θ₂")
plot!(x -> perturbed_argmax([3.0, x, 1.0]); label="perturbed argmax")
# Let's try to compute gradients:
Zygote.gradient(perturbed_argmax, θ)[1]
# Now it works, they are non zero !
plot(x -> argmax([3.0, x, 1.0]), 1:0.01:5; label="argmax", xlabel="θ₂")
plot!(x -> perturbed_argmax([3.0, x, 1.0]); label="perturbed argmax")
plot!(x -> Zygote.gradient(perturbed_argmax, [3.0, x, 1.0])[1][2]; label="gradient")

# This `perturbed_argmax` layer can now be used for instance inside a machine learning
# pipeline for multiclass classification instead of the usual softmax layer.

# ## Math behind the implementation

# TODO

# ## Specific implementations for the argmax function

# [`sparse_argmax`](@ref) and [`soft_argmax`](@ref)
