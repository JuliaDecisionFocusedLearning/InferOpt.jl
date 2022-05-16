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

function test_loop(pipelines, data_train, data_test, true_maximizer, cost, true_encoder, metrics; nb_epochs=500, show_plots=true, setting_name="???")
    for target in keys(pipelines), pipeline in pipelines[target]
        (; encoder, maximizer, loss) = pipeline
        # model = InferOptModel(encoder, maximizer, loss)

        flux_loss = define_flux_loss(encoder, maximizer, loss, target)
        train_metrics = [metric("Train $name") for (name, metric) in metrics]
        test_metrics = [metric("Test $name") for (name, metric) in metrics]
        opt = ADAM()
        additional_info = (; cost, true_encoder)
        pipeline(x) = true_maximizer(encoder(x))

        trainer = InferOptTrainer(
            encoder,
            train_metrics, test_metrics,
            opt, flux_loss, pipeline, additional_info
        )

        @info "Testing $setting_name" target encoder maximizer loss
        @showprogress for _ in 1:nb_epochs
            compute_metrics!(trainer, data_train, data_test)
            (;X, θ, Y) = data_train
            Flux.train!(trainer.flux_loss, Flux.params(trainer.encoder), zip(X, θ, Y), trainer.opt)
        end

        ## Evaluation
        if show_plots
            plot_perf(trainer)
        end
        test_perf(trainer; test_name="$target - $maximizer - $loss")
    end
    return nothing
end
