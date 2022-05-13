# ---

struct InferOptModel{E, M, L}
    encoder::E
    maximizer::M
    loss::L
end

# ---

struct InferOptDataset{X, Y, T}
    X_train::X
    X_test::X
    θ_train::T
    θ_test::T
    Y_train::Y
    Y_test::Y
end

# ---

struct InferOptTrainer{D <: InferOptDataset, M <: InferOptModel, MM, O, L, T}
    dataset::D
    model::M
    metrics::Vector{MM}
    opt::O
    flux_loss::L
    true_maximizer::T
end
#additional_info::I

# TODO: constructors

function compute_metrics!(t::InferOptTrainer)
    Y_train_pred = generate_predictions(t.model.encoder, t.true_maximizer, t.dataset.X_train)
    Y_test_pred = generate_predictions(t.model.encoder, t.true_maximizer, t.dataset.X_test)
    for metric in t.metrics
        compute_value!(metric, t; Y_train_pred, Y_test_pred)
    end
end

function get_data_train(t::InferOptTrainer)
    return (t.dataset.X_train, t.dataset.θ_train, t.dataset.Y_train)
end

function get_data_test(t::InferOptTrainer)
    return (t.dataset.X_test, t.dataset.θ_test, t.dataset.Y_test)
end
