# # Tutorial

# ## Context

#=
Let us imagine that we observe the itineraries chosen by a public transport user in several different networks, and that we want to recover their preferences (a.k.a. utility function).

More precisely, each point in our dataset consists in:
- a graph `g` & two vertices `i` and `j`
- a shortest path from `i` to `j` in `g`, computed with user-defined edge costs

Assume we know the features that the user combines to define edge costs, but we don't know the respective weights of these features.

We will use `InferOpt.jl` to learn these weights, so that we may propose relevant paths to the user in the future.
=#

# ## Setup

# We start by importing the relevant packages

using Flux
using Graphs
using InferOpt
using LinearAlgebra
using ProgressMeter
using Random
using StatsBase: mean, sample
using SparseArrays
using Test

Random.seed!(63);

#=
We first define an encoder, which takes a graph `g` as input and computes an embedding matrix `x`.
=#

dim = 5

function encoder(g::AbstractGraph)
    x = rand(dim, ne(g))
    return x
end;

#=
Now we define our maximizer, which solves the shortest path problem on `g` with edge costs `-θ`.
The minus sign is important since `InferOpt.jl` deals with linear maximization problems, while the shortest path is a linear minimization problem.
=#

function maximizer(θ; g, i, j)
    ## Build the cost matrix
    Ic = [src(e) for e in edges(g)]
    Jc = [dst(e) for e in edges(g)]
    Vc = [-θ[k] for (k, e) in enumerate(edges(g))]
    c = Symmetric(sparse(Ic, Jc, Vc, ne(g), ne(g)))
    ## Compute the shortest path from i to j
    path = a_star(g, i, j, c)
    ## Encode it as a binary vector
    Iy = Int[k for (k, e) in enumerate(edges(g)) if (e in path) || reverse(e) in path]
    Vy = ones(Int, length(Iy))
    y = sparsevec(Iy, Vy, ne(g))
    return y
end;

#=
To generate instances, we sample from a family of connected graphs and select the source and destination at random.
We then perform the following steps:
1. extract the features with our `encoder`
2. deduce edge costs with the `true_model` of the user's preferences
3. compute the shortest path with our `maximizer`
=#

function build_instance(connected_graph_generator, true_model)
    g = connected_graph_generator()
    i, j = sample(1:nv(g), 2; replace=false)
    x = encoder(g)
    θ = true_model(x)
    y = maximizer(θ; g=g, i=i, j=j)
    return (x, y, (g=g, i=i, j=j))
end;

# ## Learning

#=
Now we put everything together, starting with the true user model for edge costs.
We use a linear combination of the features, composed with the sigmoid activation and a negative absolute value.
This last step is used to obtain negative values in `θ`, which will correspond to positive edge costs in `-θ`.
=#

true_model = Chain(Dense(dim, 1), z -> -abs.(z), vec);

# We generate 200 grid graphs of size 10*10.

connected_graph_generator() = grid((10, 10))
training_data = [build_instance(connected_graph_generator, true_model) for _ in 1:200];

# We create a trainable model with the same structure as the true model but another set of randomly-initialized weights.

initial_model = Chain(Dense(dim, 1), z -> -abs.(z), vec);

#=
Here is the crucial part where `InferOpt.jl` intervenes: the choice of a clever loss function that enables us to
- differentiate through the shortest path maximizer, even though it is a discrete operation
- evaluate the quality of our model based on the paths that it recommends
=#

loss = FenchelYoungLoss(Perturbed(maximizer; ε=1.0, M=2));

# Finally, we choose a standard gradient optimizer

opt = ADAM();

# We can now train our model for 100 epochs

model = deepcopy(initial_model)
par = Flux.params(model)

for epoch in 1:100
    for sample in training_data
        x, y, kwargs = sample
        gs = gradient(par) do
            loss(model(x), y; kwargs...)
        end
        Flux.update!(opt, par, gs)
    end
end;

# ## Results

# First, we compare the learned weights with their true (hidden) values

w = model[1].weight
true_w = true_model[1].weight
vcat(w / norm(w), true_w / norm(true_w))

#=
We are already quite close to recovering the exact user weights.
But in reality, it doesn't matter as much as our ability to provide accurate path predictions.

To evaluate it, we use the Hamming distance on the space of (binary-encoded) paths.
=#

function hamming_distance(y, ȳ)
    return sum(y[i] != ȳ[i] for i in eachindex(y))
end;

# Let us now compare our predictions with the actual paths on the training set

Ȳ = [y for (x, y, kwargs) in training_data];
Y = [maximizer(model(x); kwargs...) for (x, y, kwargs) in training_data];

train_error = mean(hamming_distance(y, ȳ) / length(y) for (y, ȳ) in zip(Y, Ȳ))

# Not too bad, at least compared with our initial model.

Y0 = [maximizer(initial_model(x); kwargs...) for (x, y, kwargs) in training_data];

train_error_initial_model = mean(
    hamming_distance(y, ȳ) / length(y) for (y, ȳ) in zip(Y0, Ȳ)
)

# But we should check on a test set, just to be sure.

test_data = [build_instance(connected_graph_generator, true_model) for _ in 1:1000];

Ȳ_test = [y for (x, y, kwargs) in test_data];
Y_test = [maximizer(model(x); kwargs...) for (x, y, kwargs) in test_data];

test_error = mean(hamming_distance(y, ȳ) / length(y) for (y, ȳ) in zip(Y_test, Ȳ_test))

# Again, we compare to the initial model

Y_test0 = [maximizer(initial_model(x); kwargs...) for (x, y, kwargs) in test_data];

test_error_initial_model = mean(
    hamming_distance(y, ȳ) / length(y) for (y, ȳ) in zip(Y_test0, Ȳ_test)
)

# This is definitely a success. We just add a few tests for Continuous Integration purposes:

@test train_error < train_error_initial_model / 3

#-

@test test_error < test_error_initial_model / 3
