## Performance metrics

function init_perf(;
    train_losss=true,
    test_losss=true,
    train_errors=true,
    test_errors=true,
    train_cost_gaps=true,
    test_cost_gaps=true,
    parameter_errors=true,
)
    perf_storage = (
        train_losses=train_losss ? Float64[] : nothing,
        test_losses=test_losss ? Float64[] : nothing,
        train_errors=train_errors ? Float64[] : nothing,
        test_errors=test_errors ? Float64[] : nothing,
        train_cost_gaps=train_cost_gaps ? Float64[] : nothing,
        test_cost_gaps=test_cost_gaps ? Float64[] : nothing,
        parameter_errors=parameter_errors ? Float64[] : nothing,
    )
    return perf_storage
end

function update_perf!(
    perf_storage::NamedTuple;
    data_train,
    data_test,
    true_encoder,
    encoder,
    true_maximizer,
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

    (X_train, _, Y_train) = data_train
    (X_test, _, Y_test) = data_test

    if !isnothing(train_losses)
        train_loss = sum(flux_loss(t...) for t in zip(data_train...))
        push!(train_losses, train_loss)
    end

    if !isnothing(test_losses)
        test_loss = sum(flux_loss(t...) for t in zip(data_test...))
        push!(test_losses, test_loss)
    end

    Y_train_pred = generate_predictions(encoder, true_maximizer, X_train)
    Y_test_pred = generate_predictions(encoder, true_maximizer, X_test)

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

    w_true = first(true_encoder).weight
    w_learned = first(encoder).weight
    parameter_error = normalized_mape(w_true, w_learned)

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

function test_perf(trainer::InferOptTrainer; test_name::String)
    @testset "$test_name" begin
        for metric in trainer.train_metrics
            test_perf(metric)
        end

        for metric in trainer.test_metrics
            test_perf(metric)
        end

        # # Prediction errors
        # if length(train_errors) > 0
        #     @test train_errors[end] < train_errors[1] / 2
        # end
        # if length(test_errors) > 0
        #     @test test_errors[end] < test_errors[1] / 2
        # end
        # # Cost
        # if length(train_cost_gaps) > 0
        #     @test train_cost_gaps[end] < train_cost_gaps[1]
        # end
        # if length(test_cost_gaps) > 0
        #     @test test_cost_gaps[end] < test_cost_gaps[1]
        # end
        # # Parameter errors
        # if length(parameter_errors) > 0
        #     @test parameter_errors[end] < parameter_errors[1] / 2
        # end
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
        plt = lineplot(
            train_errors;
            xlabel="Epoch",
            title="Train error",
            # ylim=(0, maximum(train_errors)),
        )
        push!(plts, plt)
    end

    if length(test_errors) > 0
        plt = lineplot(
            test_errors; xlabel="Epoch", title="Test error",
            # ylim=(0, maximum(test_errors))
        )
        push!(plts, plt)
    end

    if length(train_cost_gaps) > 0
        plt = lineplot(
            train_cost_gaps;
            xlabel="Epoch",
            title="Train cost gap",
            # ylim=(0, maximum(train_cost_gaps)),
        )
        push!(plts, plt)
    end

    if length(train_cost_gaps) > 0
        plt = lineplot(
            test_cost_gaps;
            xlabel="Epoch",
            title="Test cost gap",
            # ylim=(0, maximum(test_cost_gaps)),
        )
        push!(plts, plt)
    end

    if length(parameter_errors) > 0
        plt = lineplot(
            parameter_errors;
            xlabel="Epoch",
            title="Parameter error",
            # ylim=(0, maximum(parameter_errors)),
        )
        push!(plts, plt)
    end

    for plt in plts
        println(plt)
    end
    return nothing
end

function plot_perf(t::InferOptTrainer)
    plts = []
    for m_list in (t.train_metrics, t.test_metrics)
        for metric in m_list
            plt = lineplot(metric.history; xlabel="Epoch", title=name(metric))
            push!(plts, plt)
        end
    end

    for plt in plts
        println(plt)
    end
    return nothing
end