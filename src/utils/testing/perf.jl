## Performance metrics

function test_perf(trainer::InferOptTrainer; test_name::String="Test")
    @testset "$test_name" begin
        for metric in trainer.train_metrics
            test_perf(metric)
        end

        for metric in trainer.test_metrics
            test_perf(metric)
        end
    end
end

function plot_perf(t::InferOptTrainer; lineplot_function::Function)
    for m_list in (t.train_metrics, t.test_metrics)
        for metric in m_list
            plt = lineplot_function(metric.history; xlabel="Epoch", title=name(metric))
            println(plt)
        end
    end

    return nothing
end
