struct InferOptDataset{D}
    train::D
    test::D
end

function InferOptDataset(;
    X_train=nothing,
    thetas_train=nothing,
    Y_train=nothing,
    X_test=nothing,
    thetas_test=nothing,
    Y_test=nothing,
)
    data_train = (X=X_train, thetas=thetas_train, Y=Y_train)
    data_test = (X=X_test, thetas=thetas_test, Y=Y_test)
    return InferOptDataset(data_train, data_test)
end

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

    data = InferOptDataset(;
        X_train=X_train,
        thetas_train=thetas_train,
        Y_train=Y_train,
        X_test=X_test,
        thetas_test=thetas_test,
        Y_test=Y_test,
    )
    return data
end
