# # Basic tutorial

# ## Context

#=
Let us imagine that we observe the itineraries chosen by a public transport user in several different networks, and that we want to understand their decision-making process (a.k.a. recover their utility function).

More precisely, each point in our dataset consists in:
- a graph $G$
- a shortest path $P$ from the top left to the bottom right corner

We don't know the true costs that were used to compute the shortest path, but we can exploit a set of features to approximate these costs.
The question is: how should we combine these features?

We will use InferOpt.jl to learn the appropriate weights, so that we may propose relevant paths to the user in the future.
=#

using Flux
using Graphs
using GridGraphs
using InferOpt
using LinearAlgebra
using ProgressMeter
using Random
using Statistics
using Test
using UnicodePlots

Random.seed!(63);

# ## Grid graphs

#=
For the purposes of this tutorial, we consider grid graphs, as implemented in [GridGraphs.jl](https://github.com/gdalle/GridGraphs.jl).
In such graphs, each vertex corresponds to a couple of coordinates $(i, j)$, where $1 \leq i \leq h$ and $1 \leq j \leq w$.

To ensure acyclicity, we only allow the user to move right, down or both.
Since the cost of a move is defined as the cost of the arrival vertex, any grid graph is entirely characterized by its cost matrix $\theta \in \mathbb{R}^{h \times w}$.
=#

h, w = 50, 100
queen_directions = GridGraphs.QUEEN_DIRECTIONS_ACYCLIC
g = GridGraph(rand(h, w); directions=queen_directions);

#=
For convenience, GridGraphs.jl also provides custom functions to compute shortest paths efficiently.
Let us see what those paths look like.
=#

p = path_to_matrix(g, grid_topological_sort(g, 1, nv(g)));
spy(p)

# ## Dataset

#=
As announced, we do not know the cost of each vertex, only a set of relevant features.
Let us assume that the user combines them using a shallow neural network.
=#

nb_features = 5
true_encoder = Chain(Dense(nb_features, 1), z -> dropdims(z; dims=1));

#=
The true vertex costs computed from this encoding are then used within shortest path computations.
To be consistent with the literature, we frame this problem as a linear maximization problem, which justifies the change of sign in front of $\theta$.
Note that `linear_maximizer` can take keyword arguments, eg. to give additional information about the instance that `θ` doesn't contain.
=#

function linear_maximizer(θ; directions)
    g = GridGraph(-θ; directions=directions)
    path = grid_topological_sort(g, 1, nv(g))
    return path_to_matrix(g, path)
end;

#=
We now have everything we need to build our dataset.
=#

nb_instances = 30

X_train = [randn(Float32, nb_features, h, w) for n in 1:nb_instances];
θ_train = [true_encoder(x) for x in X_train];
Y_train = [linear_maximizer(θ; directions=queen_directions) for θ in θ_train];

# ## Learning

#=
We create a trainable model with the same structure as the true encoder but another set of randomly-initialized weights.
=#

initial_encoder = Chain(Dense(nb_features, 1), z -> dropdims(z; dims=1));

#=
Here is the crucial part where InferOpt.jl intervenes: the choice of a clever loss function that enables us to
- differentiate through the shortest path maximizer, even though it is a combinatorial operation
- evaluate the quality of our model based on the paths that it recommends
=#

layer = PerturbedMultiplicative(linear_maximizer; ε=0.1, nb_samples=5);
loss = FenchelYoungLoss(layer);

#=
This probabilistic layer is just a thin wrapper around our `linear_maximizer`, but with a very different behavior:
=#

p_layer = layer(θ_train[1]; directions=queen_directions);
spy(p_layer)

#=
Instead of choosing just one path, it spreads over several possible paths, allowing its output to change smoothly as $\theta$ varies.
Thanks to this smoothing, we can now train our model with a standard gradient optimizer.
=#

encoder = deepcopy(initial_encoder)
opt = Flux.Adam();
opt_state = Flux.setup(opt, encoder)
losses = Float64[]
for epoch in 1:100
    l = 0.0
    for (x, y) in zip(X_train, Y_train)
        grads = Flux.gradient(encoder) do m
            l += loss(m(x), y; directions=queen_directions)
        end
        Flux.update!(opt_state, encoder, grads[1])
    end
    push!(losses, l)
end;

# ## Results

#=
Since the Fenchel-Young loss is convex, it is no wonder that optimization worked like a charm.
=#

lineplot(losses; xlabel="Epoch", ylabel="Loss")

#=
To assess performance, we can compare the learned weights with their true (hidden) values
=#

learned_weight = encoder[1].weight / norm(encoder[1].weight)
true_weight = true_encoder[1].weight / norm(true_encoder[1].weight)
hcat(learned_weight, true_weight)

#=
We are quite close to recovering the exact user weights.
But in reality, it doesn't matter as much as our ability to provide accurate path predictions.
Let us therefore compare our predictions with the actual paths on the training set.
=#

normalized_hamming(x, y) = mean(x[i] != y[i] for i in eachindex(x));

#-

Y_train_pred = [linear_maximizer(encoder(x); directions=queen_directions) for x in X_train];

train_error = mean(
    normalized_hamming(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred)
)

# Not too bad, at least compared with our random initial encoder.

Y_train_pred_initial = [
    linear_maximizer(initial_encoder(x); directions=queen_directions) for x in X_train
];

train_error_initial = mean(
    normalized_hamming(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred_initial)
)

#=
This is definitely a success.
Of course in real prediction settings we should measure performance on a test set as well.
This is left as an exercise to the reader.
=#

# CI tests, not included in the documentation  #src

@test train_error < train_error_initial / 3  #src
