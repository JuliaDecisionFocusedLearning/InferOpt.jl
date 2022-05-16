struct InferOptDataset{X, Y, T}
    X::X
    θ::T
    Y::Y
end

InferOptDataset(; X=nothing, θ=nothing, Y=nothing) = InferOptDataset(X, θ, Y)

function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
    N = length(X)
    N_train = floor(Int, N * train_percentage)
    N_test = N - N_train
    train_ind, test_ind = 1:N_train, (N_train + 1):(N_train + N_test)
    X_train, X_test = X[train_ind], X[test_ind]
    return X_train, X_test
end

function generate_dataset(
    true_encoder,
    true_maximizer;
    nb_features::Integer,
    instance_dim,
    nb_instances::Integer,
    noise_std::Real,
)
    X = [randn(nb_features, instance_dim...) for n in 1:nb_instances]
    thetas = [true_encoder(x) for x in X]
    noiseless_Y = [true_maximizer(θ) for θ in thetas]
    noisy_Y = [true_maximizer(θ + noise_std * randn(instance_dim...)) for θ in thetas]

    X_train, X_test = train_test_split(X)
    thetas_train, thetas_test = train_test_split(thetas)
    Y_train, _ = train_test_split(noisy_Y)
    _, Y_test = train_test_split(noiseless_Y)

    data_train = InferOptDataset(X_train, thetas_train, Y_train)
    data_test = InferOptDataset(X_test, thetas_test, Y_test)
    return data_train, data_test
end
