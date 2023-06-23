function test_pipeline!(
    ::Type{PL};
    instance_dim,
    true_maximizer,
    maximizer,
    loss,
    error_function,
    true_encoder=encoder_factory(),
    cost=(y; instance) -> -dot(y, true_encoder(instance)),
    maximizer_kwargs=NamedTuple(),
    loss_kwargs=NamedTuple(),
    epochs=EPOCHS,
    decrease=DECREASE,
    verbose=VERBOSE,
) where {PL<:PipelineLoss}
    data_train, data_test = generate_dataset(true_encoder, true_maximizer; instance_dim)
    (X_train, thetas_train, Y_train) = data_train

    encoder = encoder_factory()
    pipeline_loss = PL(encoder, maximizer, loss)
    opt_state = Flux.setup(Flux.Adam(), encoder)
    perf_storage = init_perf()
    
    for _ in 1:epochs
        update_perf!(
            perf_storage;
            data_train,
            data_test,
            true_encoder,
            true_maximizer,
            encoder,
            pipeline_loss,
            error_function,
            cost,
        )
        for (X, θ, y) in zip(X_train, thetas_train, Y_train)
            grads = Flux.gradient(pipeline_loss) do pl
                pl(X, θ, y; maximizer_kwargs, loss_kwargs)
            end
            Flux.update!(opt_state, encoder, grads[1])
        end
    end

    test_perf(perf_storage; decrease=decrease)
    if verbose
        @info "Testing" maximizer loss
        print_plot_perf(perf_storage)
    end
    return perf_storage
end
