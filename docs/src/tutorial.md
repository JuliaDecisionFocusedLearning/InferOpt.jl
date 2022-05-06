```@meta
EditURL = "<unknown>/test/tutorial.jl"
```

# Tutorial

## Context

Let us imagine that we observe the itineraries chosen by a public transport user in several different networks, and that we want to understand their decision-making process (a.k.a. recover their utility function).

More precisely, each point in our dataset consists in:
- a graph ``G``
- a shortest path ``P`` from the top left to the bottom right corner

We don't know the true costs that were used to compute the shortest path, but we can exploit a set of features to approximate these costs.
The question is: how should we combine these features?

We will use `InferOpt` to learn the appropriate weights, so that we may propose relevant paths to the user in the future.

````@example tutorial
using Flux
using InferOpt
using InferOpt.GridGraphs
using InferOpt.Testing
using LinearAlgebra
using ProgressMeter
using Random
using Statistics
using Test
using UnicodePlots

Random.seed!(63);
nothing #hide
````

## Grid graphs

For the purposes of this tutorial, we consider grid graphs, as implemented in `InferOpt.GridGraphs`.
In such graphs, each vertex corresponds to a couple of coordinates ``(i, j)``, where ``1 \leq i \leq h`` and ``1 \leq j \leq w``.

To ensure acyclicity, we only allow the user to move right, down or both.
Since the cost of a move is defined as the cost of the arrival vertex, any grid graph is entirely characterized by its cost matrix ``\theta \in \mathbb{R}^{h \times w}``.

````@example tutorial
h, w = 50, 100
g = AcyclicGridGraph(rand(h, w));
nothing #hide
````

For convenience, `InferOpt.GridGraphs` also provides custom functions to compute shortest paths.
Let us see what those look like.

````@example tutorial
p = grid_shortest_path(g, 1, nv(g));
spy(p)
````

## Dataset

As announced, we do not know the cost of each vertex, only a set of relevant features.
Let us assume that the user combines them using a shallow neural network.

````@example tutorial
nb_features = 5

true_encoder = Chain(Dense(nb_features, 1), dropfirstdim);
nothing #hide
````

The true vertex costs computed from this encoding are then used within shortest path computations.
To be consistent with the literature, we frame this problem as a linear maximization problem, which justifies the change of sign in front of ``\theta``.

````@example tutorial
function linear_maximizer(θ)
    g = AcyclicGridGraph(-θ)
    return grid_shortest_path(g, 1, nv(g))
end;
nothing #hide
````

We now have everything we need to build our dataset.

````@example tutorial
nb_instances = 30

X_train = [randn(nb_features, h, w) for n in 1:nb_instances];
θ_train = [true_encoder(x) for x in X_train];
Y_train = [linear_maximizer(θ) for θ in θ_train];
nothing #hide
````

## Learning

We create a trainable model with the same structure as the true encoder but another set of randomly-initialized weights.

````@example tutorial
initial_encoder = Chain(Dense(nb_features, 1), dropfirstdim);
nothing #hide
````

Here is the crucial part where `InferOpt` intervenes: the choice of a clever loss function that enables us to
- differentiate through the shortest path maximizer, even though it is a combinatorial operation
- evaluate the quality of our model based on the paths that it recommends

````@example tutorial
regularized_predictor = Perturbed(linear_maximizer; ε=1.0, M=5);
loss = FenchelYoungLoss(regularized_predictor);
nothing #hide
````

The regularized predictor is just a thin wrapper around our `linear_maximizer`, but with a very different behavior:

````@example tutorial
p_regularized = regularized_predictor(θ_train[1]);
spy(p_regularized)
````

Instead of choosing just one path, it spreads over several possible paths, allowing its output to change smoothly as ``\theta`` varies.
Thanks to this smoothing, we can now train our model with a standard gradient optimizer.

````@example tutorial
encoder = deepcopy(initial_encoder)
opt = ADAM();
losses = Float64[]
for epoch in 1:100
    l = 0.
    for (x, y) in zip(X_train, Y_train)
        grads = gradient(Flux.params(encoder)) do
            l += loss(encoder(x), y)
        end
        Flux.update!(opt, Flux.params(encoder), grads)
    end
    push!(losses, l)
end;
nothing #hide
````

## Results

Since the Fenchel-Young loss is convex, it is no wonder that optimization worked like a charm.

````@example tutorial
lineplot(losses, xlabel="Epoch", ylabel="Loss")
````

To assess performance, we can compare the learned weights with their true (hidden) values

````@example tutorial
learned_weight = encoder[1].weight / norm(encoder[1].weight)
true_weight = true_encoder[1].weight / norm(true_encoder[1].weight)
@info "Parameter error" learned_weight true_weight
````

We are quite close to recovering the exact user weights.
But in reality, it doesn't matter as much as our ability to provide accurate path predictions.
Let us therefore compare our predictions with the actual paths on the training set.

````@example tutorial
Y_train_pred = [linear_maximizer(encoder(x)) for x in X_train];

train_error = mean(
    normalized_hamming_distance(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred)
)
````

Not too bad, at least compared with our random initial encoder.

````@example tutorial
Y_train_pred_initial = [linear_maximizer(initial_encoder(x)) for x in X_train];

train_error_initial = mean(
    normalized_hamming_distance(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred_initial)
)
````

This is definitely a success.
Of course in real prediction settings we should measure performance on a test set as well.
This is left as an exercise to the reader.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

