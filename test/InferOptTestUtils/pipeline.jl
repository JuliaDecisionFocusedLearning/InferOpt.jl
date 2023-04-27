dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_positive(z::AbstractArray) = softplus.(z)

function test_pipeline!(
    pipeline,
    pipeline_loss;
    true_encoder,
    true_maximizer,
    data_train,
    data_test,
    error_function,
    cost,
    epochs,
    verbose,
    setting_name="???",
)
    (; encoder, maximizer, loss) = pipeline
    if verbose
        @info "Testing $setting_name" maximizer loss
    end

    ## Optimization
    opt = Flux.Adam()
    perf_storage = init_perf()
    prog = Progress(epochs; enabled=verbose)

    for _ in 1:epochs
        next!(prog)
        update_perf!(
            perf_storage;
            data_train=data_train,
            data_test=data_test,
            true_encoder=true_encoder,
            encoder=encoder,
            true_maximizer=true_maximizer,
            pipeline_loss=pipeline_loss,
            error_function=error_function,
            cost=cost,
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
