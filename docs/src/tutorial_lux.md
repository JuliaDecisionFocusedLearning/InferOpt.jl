```@meta
EditURL = "../../examples/tutorial_lux.jl"
```

# Basic tutorial (Lux)

## Context

Let us imagine that we observe the itineraries chosen by a public transport user in several different networks, and that we want to understand their decision-making process (a.k.a. recover their utility function).

More precisely, each point in our dataset consists in:
- a graph $G$
- a shortest path $P$ from the top left to the bottom right corner

We don't know the true costs that were used to compute the shortest path, but we can exploit a set of features to approximate these costs.
The question is: how should we combine these features?

We will use InferOpt.jl to learn the appropriate weights, so that we may propose relevant paths to the user in the future.

This tutorial uses [Lux.jl](https://github.com/LuxDL/Lux.jl) as the neural network framework.
For a Flux.jl version, see the companion `tutorial.jl`.

````@example tutorial_lux
using InferOpt
using LinearAlgebra
using Lux
using Optimisers: Adam
using Random
using Statistics
using Test
using UnicodePlots
using Zygote

rng = Random.default_rng()
Random.seed!(rng, 63);
nothing #hide
````

## Grid graphs

For the purposes of this tutorial, we consider grid graphs.
In such graphs, each vertex corresponds to a couple of coordinates $(i, j)$, where $1 \leq i \leq h$ and $1 \leq j \leq w$.

To ensure acyclicity, we only allow the user to move right or down.
Since the cost of a move is defined as the cost of the arrival vertex, any grid graph is entirely characterized by its weight matrix $\theta \in \mathbb{R}^{h \times w}$.

````@example tutorial_lux
h, w = 50, 100
θ_example = rand(h, w);
nothing #hide
````

With moves restricted to right or down, the shortest path from the top-left to the bottom-right
corner can be computed using dynamic programming.

````@example tutorial_lux
function grid_shortest_path_matrix(θ::AbstractMatrix)
    h, w = size(θ)
    dist = fill(Inf, h, w)
    prev_i = zeros(Int, h, w)
    prev_j = zeros(Int, h, w)
    dist[1, 1] = θ[1, 1]
    for j in 1:w, i in 1:h
        (i == 1 && j == 1) && continue
        for (ni, nj) in ((i - 1, j), (i, j - 1))
            if 1 <= ni && 1 <= nj
                d = dist[ni, nj] + θ[i, j]
                if d < dist[i, j]
                    dist[i, j] = d
                    prev_i[i, j] = ni
                    prev_j[i, j] = nj
                end
            end
        end
    end
    mat = zeros(h, w)
    ci, cj = h, w
    while (ci, cj) != (0, 0)
        mat[ci, cj] = 1.0
        ci, cj = prev_i[ci, cj], prev_j[ci, cj]
    end
    return mat
end

p = grid_shortest_path_matrix(θ_example);
spy(p)
````

## Dataset

As announced, we do not know the cost of each vertex, only a set of relevant features.
Let us assume that the user combines them using a shallow neural network.

In Lux, models are stateless: parameters and state are managed explicitly.
We define the encoder architecture and initialize the true encoder parameters.

````@example tutorial_lux
nb_features = 5
encoder_model = Chain(Dense(nb_features => 1), WrappedFunction(z -> dropdims(z; dims=1)))
true_ps, true_st = Lux.setup(rng, encoder_model);
nothing #hide
````

The true vertex costs computed from this encoding are then used within shortest path computations.
To be consistent with the literature, we frame this problem as a linear maximization problem,
which justifies the change of sign in front of $\theta$.

````@example tutorial_lux
function linear_maximizer(θ; kwargs...)
    return grid_shortest_path_matrix(-θ)
end;
nothing #hide
````

We now have everything we need to build our dataset.

````@example tutorial_lux
nb_instances = 30

X_train = [randn(rng, Float32, nb_features, h, w) for n in 1:nb_instances];
θ_train = [first(encoder_model(x, true_ps, true_st)) for x in X_train];
Y_train = [linear_maximizer(θ) for θ in θ_train];
nothing #hide
````

## Learning

We create a trainable model with the same structure as the true encoder but another set of randomly-initialized weights.

With Lux, `Lux.setup` returns separate parameter and state containers.
Gradients are taken with respect to the parameters only.

````@example tutorial_lux
ps, st = Lux.setup(rng, encoder_model)
initial_ps = deepcopy(ps);
nothing #hide
````

Here is the crucial part where InferOpt.jl intervenes: the choice of a clever loss function that enables us to
- differentiate through the shortest path maximizer, even though it is a combinatorial operation
- evaluate the quality of our model based on the paths that it recommends

Note that InferOpt is framework-agnostic (it uses ChainRulesCore), so its layers and losses work identically with Lux and Zygote.

````@example tutorial_lux
layer = PerturbedMultiplicative(linear_maximizer; ε=0.1, nb_samples=5);
loss = FenchelYoungLoss(layer);
nothing #hide
````

This probabilistic layer is just a thin wrapper around our `linear_maximizer`, but with a very different behavior:

````@example tutorial_lux
p_layer = layer(θ_train[1]);
spy(p_layer)
````

Instead of choosing just one path, it spreads over several possible paths, allowing its output to change smoothly as $\theta$ varies.
Thanks to this smoothing, we can now train our model with a standard gradient optimizer.

We use Lux's `Training` API, which manages parameter and optimizer state via a `TrainState`.
The objective function passed to the Training API must follow the signature
`(model, ps, st, data) -> (loss_value, updated_st, stats)`.
We wrap our InferOpt loss in a callable struct to make it compatible with this interface.

````@example tutorial_lux
struct LuxLoss{L}
    loss::L
end

function (obj::LuxLoss)(model, ps, st, (x, y))
    θ_pred, st = model(x, ps, st)
    return obj.loss(θ_pred, y), st, (;)
end

lux_loss = LuxLoss(loss)
````

````@example tutorial_lux
function train_loop(model, ps, st, lux_loss, X_train, Y_train; epochs=100)
    train_state = Training.TrainState(model, ps, st, Adam(1.0f-3))
    losses = Float64[]
    for _ in 1:epochs
        epoch_loss = 0.0
        for (x, y) in zip(X_train, Y_train)
            _, l, _, train_state = Training.single_train_step!(
                AutoZygote(), lux_loss, (x, y), train_state
            )
            epoch_loss += l
        end
        push!(losses, epoch_loss)
    end
    return train_state.parameters, losses
end

ps, losses = train_loop(encoder_model, ps, st, lux_loss, X_train, Y_train);
nothing #hide
````

## Results

Since the Fenchel-Young loss is convex, it is no wonder that optimization worked like a charm.

````@example tutorial_lux
lineplot(losses; xlabel="Epoch", ylabel="Loss")
````

To assess performance, we can compare the learned weights with their true (hidden) values.

With Lux, parameters live in a named tuple. We access the first layer's weight via `ps.layer_1.weight`.

````@example tutorial_lux
learned_weight = ps.layer_1.weight / norm(ps.layer_1.weight)
true_weight = true_ps.layer_1.weight / norm(true_ps.layer_1.weight)
vcat(learned_weight, true_weight)
````

We are quite close to recovering the exact user weights.
But in reality, it doesn't matter as much as our ability to provide accurate path predictions.
Let us therefore compare our predictions with the actual paths on the training set.

````@example tutorial_lux
normalized_hamming(x, y) = mean(x[i] != y[i] for i in eachindex(x));
nothing #hide
````

````@example tutorial_lux
Y_train_pred = [linear_maximizer(first(encoder_model(x, ps, st))) for x in X_train];

train_error = mean(
    normalized_hamming(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred)
)
````

Not too bad, at least compared with our random initial encoder.

````@example tutorial_lux
Y_train_pred_initial = [
    linear_maximizer(first(encoder_model(x, initial_ps, st))) for x in X_train
];

train_error_initial = mean(
    normalized_hamming(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred_initial)
)
````

This is definitely a success.
Of course in real prediction settings we should measure performance on a test set as well.
This is left as an exercise to the reader.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

