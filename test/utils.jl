using Flux
using InferOpt
using InferOpt.Testing
using ProgressMeter
using UnicodePlots

dropfirstdim(z::AbstractArray) = dropdims(z; dims=1)
make_positive(z::AbstractArray) = softplus.(z)

function test_loop(
    pipelines;
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
    pipelines = deepcopy(pipelines)

    for target in keys(pipelines), pipeline in pipelines[target]
        (; encoder, maximizer, loss) = pipeline
        pipeline_loss = define_pipeline_loss(encoder, maximizer, loss, target)
        @info "Testing $setting_name" target encoder maximizer loss

        ## Optimization

        opt = ADAM()
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
            plts = plot_perf(perf_storage; lineplot_function=lineplot)
            for plt in plts
                println(plt)
            end
        end
        test_perf(perf_storage; test_name="$target - $maximizer - $loss")
    end
end
