dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)

## Dataset

function generate_dataset(
    true_model,
    optimizer;
    nb_features::Integer,
    instance_dim::Integer,
    nb_instances::Integer,
    noise_std::Real,
)
    X = [randn(nb_features, instance_dim) for n in 1:nb_instances]
    thetas = [true_model(x) for x in X]
    noiseless_Y = [optimizer(θ) for θ in thetas]
    noisy_Y = [optimizer(θ + noise_std * randn(instance_dim)) for θ in thetas]

    X_train, X_test = InferOpt.train_test_split(X)
    thetas_train, thetas_test = InferOpt.train_test_split(thetas)
    Y_train, _ = InferOpt.train_test_split(noisy_Y)
    _, Y_test = InferOpt.train_test_split(noiseless_Y)

    data_train = (X_train, thetas_train, Y_train)
    data_test = (X_test, thetas_test, Y_test)
    return data_train, data_test
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

function define_flux_loss(model, loss, target)
    flux_loss_none(x, θ, y) = loss(model(x); instance=x)
    flux_loss_θ(x, θ, y) = loss(model(x), θ)
    flux_loss_y(x, θ, y) = loss(model(x), y)
    flux_loss_θy(x, θ, y) = loss(model(x), θ, y)

    flux_losses = Dict(
        "none" => flux_loss_none,
        "θ" => flux_loss_θ,
        "y" => flux_loss_y,
        "(θ,y)" => flux_loss_θy,
    )

    return flux_losses[target]
end

## Performance metrics

function init_perf()
    perf_storage = (
        train_losses=Float64[],
        test_losses=Float64[],
        train_errors=Float64[],
        test_errors=Float64[],
        train_cost_gaps=Float64[],
        test_cost_gaps=Float64[],
        parameter_errors=Float64[],
    )
    return perf_storage
end

function update_perf!(
    perf_storage::NamedTuple;
    data_train,
    data_test,
    true_model,
    model,
    optimizer,
    flux_loss,
    error_function,
    cost,
)
    (;
        train_losses,
        test_losses,
        train_errors,
        test_errors,
        train_cost_gaps,
        test_cost_gaps,
        parameter_errors,
    ) = perf_storage

    (X_train, thetas_train, Y_train) = data_train
    (X_test, thetas_test, Y_test) = data_test

    train_loss = sum(flux_loss(t...) for t in zip(data_train...))
    test_loss = sum(flux_loss(t...) for t in zip(data_test...))

    Y_train_pred = generate_predictions(model, optimizer, X_train)
    Y_test_pred = generate_predictions(model, optimizer, X_test)

    train_error = mean(
        error_function(y, y_pred) for (y, y_pred) in zip(Y_train, Y_train_pred)
    )
    test_error = mean(error_function(y, y_pred) for (y, y_pred) in zip(Y_test, Y_test_pred))

    train_cost = [cost(y; instance=x) for (x, y) in zip(X_train, Y_train_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(X_train, Y_train)]
    test_cost = [cost(y; instance=x) for (x, y) in zip(X_test, Y_test_pred)]
    test_cost_opt = [cost(y; instance=x) for (x, y) in zip(X_test, Y_test)]

    train_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    )
    test_cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(test_cost, test_cost_opt)
    )

    w_true = first(true_model).weight
    w_learned = first(model).weight
    parameter_error = InferOpt.normalized_mape(w_true, w_learned)

    push!(train_losses, train_loss)
    push!(test_losses, test_loss)
    push!(train_errors, train_error)
    push!(test_errors, test_error)
    push!(train_cost_gaps, train_cost_gap)
    push!(test_cost_gaps, test_cost_gap)
    push!(parameter_errors, parameter_error)
    return nothing
end

function test_perf(perf_storage::NamedTuple; test_name::String)
    (;
        train_losses,
        test_losses,
        train_errors,
        test_errors,
        train_cost_gaps,
        test_cost_gaps,
        parameter_errors,
    ) = perf_storage

    @testset "$test_name" begin
        # Losses
        if length(train_losses) > 0
            @test train_losses[end] < train_losses[1]
        end
        if length(test_losses) > 0
            @test test_losses[end] < test_losses[1]
        end
        # Prediction errors
        if length(train_errors) > 0
            @test train_errors[end] < train_errors[1] / 2
        end
        if length(test_errors) > 0
            @test test_errors[end] < test_errors[1] / 2
        end
        # Cost
        if length(train_cost_gaps) > 0
            @test train_cost_gaps[end] < train_cost_gaps[1]
        end
        if length(test_cost_gaps) > 0
            @test test_cost_gaps[end] < test_cost_gaps[1]
        end
        # Parameter errors
        if length(parameter_errors) > 0
            @test parameter_errors[end] < parameter_errors[1] / 2
        end
    end
end

function plot_perf(perf_storage::NamedTuple)
    (;
        train_losses,
        test_losses,
        train_errors,
        test_errors,
        train_cost_gaps,
        test_cost_gaps,
        parameter_errors,
    ) = perf_storage
    plts = []

    if length(train_losses) > 0
        plt = lineplot(train_losses; xlabel="Epoch", title="Train loss")
        push!(plts, plt)
    end

    if length(test_losses) > 0
        plt = lineplot(test_losses; xlabel="Epoch", title="Test loss")
        push!(plts, plt)
    end

    if length(train_errors) > 0
        plt = lineplot(train_errors; xlabel="Epoch", title="Train error")
        push!(plts, plt)
    end

    if length(test_errors) > 0
        plt = lineplot(test_errors; xlabel="Epoch", title="Test error")
        push!(plts, plt)
    end

    if length(train_cost_gaps) > 0
        plt = lineplot(train_cost_gaps; xlabel="Epoch", title="Train cost gap")
        push!(plts, plt)
    end

    if length(train_cost_gaps) > 0
        plt = lineplot(test_cost_gaps; xlabel="Epoch", title="Test cost gap")
        push!(plts, plt)
    end

    if length(parameter_errors) > 0
        plt = lineplot(
            parameter_errors;
            xlabel="Epoch",
            title="Parameter error",
            ylim=(0, maximum(parameter_errors)),
        )
        push!(plts, plt)
    end

    for plt in plts
        println(plt)
    end
    return nothing
end
