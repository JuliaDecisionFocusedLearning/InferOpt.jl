dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_positive(z::AbstractArray) = softplus.(z)

function test_pipeline!(;
    instance_dim,
    pipeline_loss,
    true_encoder,
    encoder,
    true_maximizer,
    maximizer,
    loss,
    error_function,
    cost,
    epochs=EPOCHS,
    verbose=false,
    setting_name="???",
)
    if verbose
        @info "Testing $setting_name" maximizer loss
    end

    ## Data generation
    data_train, data_test = generate_dataset(true_encoder, true_maximizer; instance_dim)

    ## Optimization
    opt = Flux.Adam()
    perf_storage = init_perf()
    prog = Progress(epochs; enabled=verbose)

    for _ in 1:epochs
        next!(prog)
        update_perf!(
            perf_storage;
            data_train,
            data_test,
            true_encoder,
            encoder,
            true_maximizer,
            pipeline_loss,
            error_function,
            cost,
        )
        Flux.train!(pipeline_loss, Flux.params(encoder), zip(data_train...), opt)
    end

    ## Evaluation
    if verbose
        plts = plot_perf(perf_storage)
        for plt in plts
            println(plt)
        end
    end
    test_perf(perf_storage; test_name="$setting_name - $maximizer - $loss")
    return perf_storage
end
