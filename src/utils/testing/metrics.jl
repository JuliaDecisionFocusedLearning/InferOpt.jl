abstract type AbstractScalarMetric end

function compute_value!(m::AbstractScalarMetric, t::InferOptTrainer, data; kwargs...)
    push!(m.history, m(t, data; kwargs...))
end

function test_perf(metric::AbstractScalarMetric)
    @test metric.history[end] < metric.history[1]
end

## ---

struct Loss <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

Loss(name="Loss") = Loss(name, Float64[])

function (m::Loss)(trainer::InferOptTrainer, data; kwargs...)
    (; X, θ, Y) = data
    return sum(trainer.flux_loss(t...) for t in zip(X, θ, Y))
end

## ---

struct HammingDistance <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

HammingDistance(name="Hamming distance") = HammingDistance(name, Float64[])

function (m::HammingDistance)(trainer::InferOptTrainer, data; Y_pred, kwargs...)
    dist = mean(
        hamming_distance(y, y_pred) for (y, y_pred) in zip(data.Y, Y_pred)
    )
    return dist
end

function test_perf(metric::HammingDistance)
    @test metric.history[end] < metric.history[1] / 2
end

## ---

struct CostGap <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

CostGap(name="Cost gap") = CostGap(name, Float64[])

function (m::CostGap)(trainer::InferOptTrainer, data; Y_pred, kwargs...)
    (; cost) = trainer.additional_info
    train_cost = [cost(y; instance=x) for (x, y) in zip(data.X, Y_pred)]
    train_cost_opt = [cost(y; instance=x) for (x, y) in zip(data.X, data.Y)]

    cost_gap = mean(
        (c - c_opt) / abs(c_opt) for (c, c_opt) in zip(train_cost, train_cost_opt)
    )
    return cost_gap
end

## ---

struct ParameterError <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

ParameterError(name="Parameter error") = ParameterError(name, Float64[])

function (m::ParameterError)(trainer::InferOptTrainer, data; kwargs...)
    (; true_encoder) = trainer.additional_info
    w_true = first(true_encoder).weight
    w_learned = first(trainer.model.encoder).weight
    parameter_error = normalized_mape(w_true, w_learned)
    return parameter_error
end

function test_perf(metric::ParameterError)
    @test metric.history[end] < metric.history[1] / 2
end

## ---

struct MeanSquaredError <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

MeanSquaredError(name="Mean squared error") = MeanSquaredError(name, Float64[])

function (m::MeanSquaredError)(trainer::InferOptTrainer, data; Y_pred, kwargs...)
    train_error = mean(
        sum((y - y_pred) .^ 2) for (y, y_pred) in zip(data.Y, Y_pred)
    )
    return train_error
end

function test_perf(metric::MeanSquaredError)
    @test metric.history[end] < metric.history[1] / 2
end

## ----

struct ScalarMetric{R <: Real, F} <: AbstractScalarMetric
    name::String
    history::Vector{R}
    f::F
end

function (m::ScalarMetric)(t::InferOptTrainer)
    return m.f(t)
end

function name(m::AbstractScalarMetric)
    return m.name
end
