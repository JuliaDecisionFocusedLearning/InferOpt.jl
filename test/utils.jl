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

function train!(t::InferOptTrainer, nb_epochs::Integer)
    @showprogress for _ in 1:nb_epochs
        compute_metrics!(t)
        (;X, θ, Y) = t.data_train
        Flux.train!(t.flux_loss, Flux.params(t.model.encoder), zip(X, θ, Y), t.opt)
    end
end

function test_loop(pipelines, data_train, data_test, true_maximizer, cost, true_encoder; nb_epochs=500, show_plots=true, setting_name="???")
    for target in keys(pipelines), pipeline in pipelines[target]
        (; encoder, maximizer, loss) = pipeline
        model = InferOptModel(encoder, maximizer, loss)

        flux_loss = define_flux_loss(encoder, maximizer, loss, target)
        train_metrics = [Loss("Train loss"), HammingDistance("Train hamming distance"), CostGap("Train cost gap"), ParameterError("Train parameter error")]
        test_metrics = [Loss("Test loss"), HammingDistance("Test Hamming distance"), CostGap("Test cost gap"), ParameterError("Test parameter error")]
        opt = ADAM()

        trainer = InferOptTrainer(
            data_train, data_test,
            model,
            train_metrics, test_metrics,
            opt, flux_loss, true_maximizer, cost, true_encoder
        )

        @info "Testing $setting_name" target encoder maximizer loss
        train!(trainer, nb_epochs)

        ## Evaluation
        if show_plots
            plot_perf(trainer)
        end
        test_perf(trainer; test_name="$target - $maximizer - $loss")
    end
    return nothing
end
