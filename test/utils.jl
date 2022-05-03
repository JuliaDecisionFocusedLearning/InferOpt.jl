## Useful functions

mape(x1, x2) = 100 * mean(abs.((x1 .- x2) ./ x1))
normalized_mape(x1, x2) = mape(x1 / norm(x1), x2 / norm(x2))

function argmax_optimizer(θ::AbstractVector; instance=nothing)
    return one_hot_argmax(θ)
end

function ranking(θ::AbstractVector; rev::Bool=false, instance=nothing)
    return invperm(sortperm(θ; rev=rev))
end

function hamming_distance(x::AbstractVector, y::AbstractVector)
    return sum(x[i] != y[i] for i in eachindex(x))
end

function normalized_hamming_distance(x::AbstractVector, y::AbstractVector)
    return mean(x[i] != y[i] for i in eachindex(x))
end

## Data generation

function generate_dataset(
    model, optimizer; N::Integer, dim_x::Integer, dim_y::Integer, σ::Real
)
    X = [randn(dim_x) for n in 1:N]
    thetas = [model(X[n]) + σ * randn(dim_y) for n in 1:N]
    Y = [optimizer(θ) for θ in thetas]
    return (X=X, thetas=thetas, Y=Y)
end

function generate_predictions(model, optimizer, X)
    Y_pred = [optimizer(model(x)) for x in X]
    return Y_pred
end

function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
    N = length(X)
    N_train = floor(Int, N * train_percentage)
    N_test = N - N_train
    train_ind, test_ind = 1:N_train, (N_train + 1):(N_train + N_test)
    X_train, X_test = X[train_ind], X[test_ind]
    return X_train, X_test
end

function plot_results(training_losses, test_losses, test_accuracies, parameter_errors)
    println(lineplot(training_losses; xlabel="Epoch", title="Training loss"))
    println(lineplot(test_losses; xlabel="Epoch", title="Test loss"))
    println(
        lineplot(test_accuracies; xlabel="Epoch", title="Test accuracy", ylim=(0, 1))
    )
    println(
        lineplot(
            parameter_errors;
            xlabel="Epoch",
            title="Parameter error",
            ylim=(0, maximum(parameter_errors)),
        ),
    )
    return nothing
end

function plot_results(training_losses, test_losses, test_accuracies, parameter_errors, training_objective_gap, test_objective_gap)
    plot_results(training_losses, test_losses, test_accuracies, parameter_errors)
    println(lineplot(training_objective_gap; xlabel="Epoch", title="Train objective gap"))
    println(lineplot(test_objective_gap; xlabel="Epoch", title="Test objective gap"))
    return nothing
end
