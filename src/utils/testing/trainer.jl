struct InferOptModel{E, M, L}
    encoder::E
    maximizer::M
    loss::L
end

struct InferOptTrainer{D <: InferOptDataset, M <: InferOptModel, MM, O, L, T, I}
    data_train::D
    data_test::D
    model::M
    train_metrics::Vector{MM}
    test_metrics::Vector{MM}
    opt::O
    flux_loss::L
    true_maximizer::T
    additional_info::I
end

# TODO: better constructor

function compute_metrics!(t::InferOptTrainer)
    Y_train_pred = generate_predictions(t.model.encoder, t.true_maximizer, t.data_train.X)
    for metric in t.train_metrics
        compute_value!(metric, t, t.data_train; Y_pred=Y_train_pred)
    end

    Y_test_pred = generate_predictions(t.model.encoder, t.true_maximizer, t.data_test.X)
    for metric in t.test_metrics
        compute_value!(metric, t, t.data_test; Y_pred=Y_test_pred)
    end
end
