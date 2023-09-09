dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_positive(z::AbstractArray) = softplus.(z)

function test_pipeline!(
    pl::PipelineLoss;
    instance_dim,
    true_maximizer,
    maximizer,
    loss,
    error_function,
    true_encoder=encoder_factory(),
    cost=(y; instance) -> -dot(y, true_encoder(instance)),
    epochs=EPOCHS,
    decrease=DECREASE,
    verbose=VERBOSE,
)
    data_train, data_test = generate_dataset(true_encoder, true_maximizer; instance_dim)

    encoder = encoder_factory()
    opt_state = Flux.setup(Flux.Adam(), encoder)
    perf_storage = init_perf()

    for ep in 1:epochs
        update_perf!(
            pl,
            perf_storage;
            data_train,
            data_test,
            true_encoder,
            true_maximizer,
            maximizer,
            encoder,
            loss,
            error_function,
            cost,
        )
        for (x, θ, y) in zip(data_train...)
            _, grads = Flux.withgradient(encoder) do _encoder
                get_loss(pl, loss, maximizer(_encoder(x)), x, θ, y)
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
