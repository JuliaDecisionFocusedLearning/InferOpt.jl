using Flux
using ProgressMeter

function define_pipeline_loss(encoder, maximizer, loss, target)
    pipeline_loss_none(x, θ, y) = loss(maximizer(encoder(x)); instance=x)
    pipeline_loss_θ(x, θ, y) = loss(maximizer(encoder(x)), θ)
    pipeline_loss_y(x, θ, y) = loss(maximizer(encoder(x)), y)
    pipeline_loss_θy(x, θ, y) = loss(maximizer(encoder(x)), θ, y)

    pipeline_losses = Dict(
        "none" => pipeline_loss_none,
        "θ" => pipeline_loss_θ,
        "y" => pipeline_loss_y,
        "(θ,y)" => pipeline_loss_θy,
    )

    return pipeline_losses[target]
end

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
    @info "Testing $setting_name" maximizer loss

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
        plts = plot_perf(perf_storage)
        for plt in plts
            println(plt)
        end
    end
    return test_perf(perf_storage; test_name="$setting_name - $maximizer - $loss")
end
