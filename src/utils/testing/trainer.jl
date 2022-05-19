struct InferOptTrainer{M, P, L, I}
    train_metrics::Vector{M}
    test_metrics::Vector{M}
    pipeline::P
    loss::L
    extra_info::I
end

function InferOptTrainer(; metrics_dict::Dict, loss, pipeline, extra_info)
    train_metrics = [metric("Train $name") for (name, metric) in metrics_dict]
    test_metrics = [metric("Test $name") for (name, metric) in metrics_dict]
    return InferOptTrainer(train_metrics, test_metrics, pipeline, loss, extra_info)
end
    I

function generate_predictions(trainer::InferOptTrainer, X)
    Y_pred = [trainer.pipeline(x) for x in X]
    return Y_pred
end

function compute_metrics!(trainer::InferOptTrainer, data::InferOptDataset; logger=nothing)
    Y_train_pred = generate_predictions(trainer, data.train.X)
    for (idx, metric) in enumerate(trainer.train_metrics)
        compute_value!(metric, trainer, data.train; Y_pred=Y_train_pred)
        log_last_measure!(metric, logger; train=true, step_increment=(idx==1 ? 1 : 0))
    end

    Y_test_pred = generate_predictions(trainer, data.test.X)
    for metric in trainer.test_metrics
        compute_value!(metric, trainer, data.test; Y_pred=Y_test_pred)
        log_last_measure!(metric, logger; train=false, step_increment=0)
    end
end
