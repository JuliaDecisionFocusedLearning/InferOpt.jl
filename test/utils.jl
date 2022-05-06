using Flux
using InferOpt
using InferOpt.Testing
using ProgressMeter

function list_standard_pipelines(true_maximizer; nb_features, cost=nothing)
    pipelines = Dict{String,Vector}()

    pipelines["θ"] = [(
        encoder=Chain(Dense(nb_features, 1), dropfirstdim),
        maximizer=identity,
        loss=SPOPlusLoss(true_maximizer),
    )]

    pipelines["(θ,y)"] = [(
        encoder=Chain(Dense(nb_features, 1), dropfirstdim),
        maximizer=identity,
        loss=SPOPlusLoss(true_maximizer),
    )]

    pipelines["y"] = [
        # Perturbations
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=FenchelYoungLoss(Perturbed(true_maximizer; ε=1.0, M=5)),
        ),
        (
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=Perturbed(true_maximizer; ε=1.0, M=5),
            loss=Flux.Losses.mse,
        ),
    ]

    if !isnothing(cost)
        pipelines["none"] = [(
            encoder=Chain(Dense(nb_features, 1), dropfirstdim),
            maximizer=identity,
            loss=PerturbedCost(true_maximizer, cost; ε=1.0, M=5),
        )]
    end

    return pipelines
end

function test_loop(
    pipelines;
    true_encoder,
    true_maximizer,
    data_train,
    data_test,
    error_function,
    cost,
    epochs,
    show_plots,
    setting_name="???",
)
    pipelines = deepcopy(pipelines)

    for target in keys(pipelines), pipeline in pipelines[target]
        (; encoder, maximizer, loss) = pipeline
        flux_loss = define_flux_loss(encoder, maximizer, loss, target)
        @info "Testing $setting_name" target encoder maximizer loss

        ## Optimization

        opt = ADAM()
        perf_storage = init_perf()

        @showprogress for _ in 1:epochs
            update_perf!(
                perf_storage;
                data_train=data_train,
                data_test=data_test,
                true_encoder=true_encoder,
                encoder=encoder,
                true_maximizer=true_maximizer,
                flux_loss=flux_loss,
                error_function=error_function,
                cost=cost,
            )
            Flux.train!(flux_loss, Flux.params(encoder), zip(data_train...), opt)
        end

        ## Evaluation

        if show_plots
            plot_perf(perf_storage)
        end
        test_perf(perf_storage; test_name="$target - $maximizer - $loss")
    end
end
