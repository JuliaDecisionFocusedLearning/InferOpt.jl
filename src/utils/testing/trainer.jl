struct InferOptModel{E, M, L}
    encoder::E
    maximizer::M
    loss::L
end

InferOptModel(;encoder, loss) = InferOptModel(encoder, identity, loss)

function (m::InferOptModel)(x::AbstractArray)
    return m.encoder(m.maximizer(x))
end

struct InferOptTrainer{E, M, O, L, P, I}
    encoder::E
    train_metrics::Vector{M}
    test_metrics::Vector{M}
    opt::O
    flux_loss::L
    pipeline::P
    # true_maximizer::T
    additional_info::I
end

function InferOptTrainer(; encoder, metrics_dict::Dict, opt, flux_loss, pipeline, additional_info)
    train_metrics = [metric("Train $name") for (name, metric) in metrics_dict]
    test_metrics = [metric("Test $name") for (name, metric) in metrics_dict]
    return InferOptTrainer(encoder, train_metrics, test_metrics, opt, flux_loss, pipeline, additional_info)
end
    I

function generate_predictions(t::InferOptTrainer, X)
    Y_pred = [t.pipeline(x) for x in X]
    return Y_pred
end

function compute_metrics!(t::InferOptTrainer, data_train, data_test; logger=nothing)
    Y_train_pred = generate_predictions(t, data_train.X)
    for (idx, metric) in enumerate(t.train_metrics)
        compute_value!(metric, t, data_train; Y_pred=Y_train_pred)
        log_last_measure!(metric, logger; train=true, step_increment=(idx==1 ? 1 : 0))
    end

    Y_test_pred = generate_predictions(t, data_test.X)
    for metric in t.test_metrics
        compute_value!(metric, t, data_test; Y_pred=Y_test_pred)
        log_last_measure!(metric, logger; train=false, step_increment=0)
    end
end
