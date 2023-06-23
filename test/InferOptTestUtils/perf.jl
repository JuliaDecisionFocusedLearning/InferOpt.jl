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

function (; encoder, encoder_ps, encoder_st, maximizer, X)
    Y_pred = [maximizer(encoder(x, encoder_ps, encoder_st)[1]) for x in X]
    return Y_pred
end

function update_perf!(
    perf_storage::NamedTuple;
    data_train,
    data_test,
    encoder,
    encoder_ps,
    encoder_st,
    true_encoder_ps,
    true_encoder_st,
    true_maximizer,
    pipeline_loss,
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

    train_loss = sum(pipeline_loss(x, θ, y) for (x, θ, y) in zip(data_train...))
    test_loss = sum(pipeline_loss(x, θ, y) for (x, θ, y) in zip(data_test...))

    Y_train_pred = generate_predictions(encoder, true_maximizer, X_train)
    Y_test_pred = generate_predictions(encoder, true_maximizer, X_test)

    train_error = mean(
        error_function(y_pred, y) for (y, y_pred) in zip(Y_train, Y_train_pred)
    )
    test_error = mean(error_function(y_pred, y) for (y, y_pred) in zip(Y_test, Y_test_pred))

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

    w_true = first(true_encoder).weight
    w_learned = first(encoder).weight
    parameter_error = normalized_mape(w_true, w_learned)

    push!(train_losses, train_loss)
    push!(test_losses, test_loss)
    push!(train_errors, train_error)
    push!(test_errors, test_error)
    push!(train_cost_gaps, train_cost_gap)
    push!(test_cost_gaps, test_cost_gap)
    push!(parameter_errors, parameter_error)
    return nothing
end

function test_perf(perf_storage::NamedTuple; decrease::Real)
    (;
        train_losses,
        test_losses,
        train_errors,
        test_errors,
        train_cost_gaps,
        test_cost_gaps,
        parameter_errors,
    ) = perf_storage

    # Losses
    if length(train_losses) > 0
        @test train_losses[end] < train_losses[1]
    end
    if length(test_losses) > 0
        @test test_losses[end] < test_losses[1]
    end
    # Prediction errors
    if length(train_errors) > 0
        @test train_errors[end] < decrease * train_errors[1]
    end
    if length(test_errors) > 0
        @test test_errors[end] < decrease * test_errors[1]
    end
    # Cost
    if length(train_cost_gaps) > 0
        @test train_cost_gaps[end] < decrease * train_cost_gaps[1]
    end
    if length(test_cost_gaps) > 0
        @test test_cost_gaps[end] < decrease * test_cost_gaps[1]
    end
    # Parameter errors
    if length(parameter_errors) > 0
        @test parameter_errors[end] < decrease * parameter_errors[1]
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

    if any(!isnan, train_losses)
        plt = lineplot(train_losses; xlabel="Epoch", title="Train loss")
        push!(plts, plt)
    end

    if any(!isnan, test_losses)
        plt = lineplot(test_losses; xlabel="Epoch", title="Test loss")
        push!(plts, plt)
    end

    if any(!isnan, train_errors)
        plt = lineplot(
            train_errors;
            xlabel="Epoch",
            title="Train error",
            # ylim=(0, maximum(train_errors)),
        )
        push!(plts, plt)
    end

    if any(!isnan, test_errors)
        plt = lineplot(
            test_errors;
            xlabel="Epoch",
            title="Test error",
            # ylim=(0, maximum(test_errors))
        )
        push!(plts, plt)
    end

    if any(!isnan, train_cost_gaps)
        plt = lineplot(
            train_cost_gaps;
            xlabel="Epoch",
            title="Train cost gap",
            # ylim=(0, maximum(train_cost_gaps)),
        )
        push!(plts, plt)
    end

    if any(!isnan, train_cost_gaps)
        plt = lineplot(
            test_cost_gaps;
            xlabel="Epoch",
            title="Test cost gap",
            # ylim=(0, maximum(test_cost_gaps)),
        )
        push!(plts, plt)
    end

    if any(!isnan, parameter_errors)
        plt = lineplot(
            parameter_errors;
            xlabel="Epoch",
            title="Parameter error",
            # ylim=(0, maximum(parameter_errors)),
        )
        push!(plts, plt)
    end

    return plts
end

function print_plot_perf(perf_storage)
    plts = plot_perf(perf_storage)
    for plt in plts
        println(plt)
    end
end
