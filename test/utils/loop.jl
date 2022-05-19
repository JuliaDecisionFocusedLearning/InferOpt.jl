using Flux
using InferOpt
using InferOpt.Testing
using ProgressMeter
using UnicodePlots

function test_loop(
    pipelines, data, true_maximizer, cost, true_encoder, metrics;
    nb_epochs=500, show_plots=true, setting_name="???"
)
    for target in keys(pipelines), pipeline in pipelines[target]
        (; encoder, maximizer, loss) = pipeline

        pipeline_loss = define_pipeline_loss(encoder, maximizer, loss, target)
        opt = ADAM()
        extra_info = (; cost, true_encoder, encoder)
        pipeline(x) = true_maximizer(encoder(x))

        trainer = InferOptTrainer(
            metrics_dict=metrics,
            loss=pipeline_loss,
            pipeline=pipeline,
            extra_info=extra_info
        )

        @info "Testing $setting_name" target encoder maximizer loss
        @showprogress for _ in 1:nb_epochs
            compute_metrics!(trainer, data)
            (;X, thetas, Y) = data.train
            Flux.train!(trainer.loss, Flux.params(encoder), zip(X, thetas, Y), opt)
        end

        ## Evaluation
        if show_plots
            plot_perf(trainer; lineplot_function=lineplot)
        end
        test_perf(trainer; test_name="$target - $maximizer - $loss")
    end
    return nothing
end
