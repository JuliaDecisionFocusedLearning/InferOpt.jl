dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_positive(z::AbstractArray) = softplus.(z)

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

    encoder = encoder_factory()
    pipeline_loss = PL(encoder, maximizer, loss, maximizer_kwargs, loss_kwargs)
    opt = Flux.Adam()
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
        Flux.train!(pipeline_loss, Flux.params(encoder), zip(data_train...), opt)
    end

    test_perf(perf_storage; decrease=decrease)
    if verbose
        @info "Testing" maximizer loss
        print_plot_perf(perf_storage)
    end
    return perf_storage
end
