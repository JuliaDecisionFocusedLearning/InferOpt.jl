function encoder_factory(nb_features=NB_FEATURES)
    return Chain(Dense(nb_features, 1), dropfirstdim, make_positive)
end

function generate_dataset(
    true_encoder,
    true_maximizer;
    instance_dim,
    nb_features::Integer=NB_FEATURES,
    nb_instances::Integer=NB_INSTANCES,
    noise_std::Real=NOISE_STD,
)
    X = [randn(Float32, nb_features, instance_dim...) for n in 1:nb_instances]
    thetas = [true_encoder(x) for x in X]
    noiseless_Y = [true_maximizer(θ; instance=x) for (x, θ) in zip(X, thetas)]
    noisy_Y = [
        true_maximizer(θ + noise_std * randn(instance_dim...); instance=x) for
        (x, θ) in zip(X, thetas)
    ]

    X_train, X_test = train_test_split(X)
    thetas_train, thetas_test = train_test_split(thetas)
    Y_train, _ = train_test_split(noisy_Y)
    _, Y_test = train_test_split(noiseless_Y)

    data_train = (X_train, thetas_train, Y_train)
    data_test = (X_test, thetas_test, Y_test)
    return data_train, data_test
end

# function generate_predictions(encoder, maximizer, X)
#     Y_pred = [maximizer(encoder(x); instance=x) for x in X]
#     return Y_pred
# end

function train_test_split(X::AbstractVector, train_percentage::Real=0.5)
    N = length(X)
    N_train = floor(Int, N * train_percentage)
    N_test = N - N_train
    train_ind, test_ind = 1:N_train, (N_train + 1):(N_train + N_test)
    X_train, X_test = X[train_ind], X[test_ind]
    return X_train, X_test
end
