using Flux
using InferOpt
using InferOpt.Testing
using ProgressMeter
using UnicodePlots

function test_loop(pipelines, data_train, data_test, true_maximizer, cost, true_encoder, metrics; nb_epochs=500, show_plots=true, setting_name="???")
    for target in keys(pipelines), pipeline in pipelines[target]
        (; encoder, maximizer, loss) = pipeline

        flux_loss = define_pipeline_loss(encoder, maximizer, loss, target)
        opt = ADAM()
        additional_info = (; cost, true_encoder)
        pipeline(x) = true_maximizer(encoder(x))

        trainer = InferOptTrainer(
            encoder=encoder,
            metrics_dict=metrics,
            opt=opt,
            flux_loss=flux_loss,
            pipeline=pipeline,
            additional_info=additional_info
        )

        @info "Testing $setting_name" target encoder maximizer loss
        @showprogress for _ in 1:nb_epochs
            compute_metrics!(trainer, data_train, data_test)
            (;X, θ, Y) = data_train
            Flux.train!(trainer.flux_loss, Flux.params(trainer.encoder), zip(X, θ, Y), trainer.opt)
        end

        ## Evaluation
        if show_plots
            plot_perf(trainer; lineplot_function=lineplot)
        end
        test_perf(trainer; test_name="$target - $maximizer - $loss")
    end
    return nothing
end
