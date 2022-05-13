abstract type AbstractScalarMetric end

function compute_value!(m::AbstractScalarMetric, t::InferOptTrainer; kwargs...)
    push!(m.history, m(t; kwargs...))
end

function test_perf(metric::AbstractScalarMetric)
    @test metric.history[end] < metric.history[1]
end

## ---

struct TrainLoss <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

TrainLoss(name="Train loss") = TrainLoss(name, Float64[])

function (m::TrainLoss)(trainer::InferOptTrainer; kwargs...)
    return sum(trainer.flux_loss(t...) for t in zip(get_data_train(trainer)...))
end

## ---

struct TestLoss <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

TestLoss(name="Test loss") = TestLoss(name, Float64[])

function (m::TestLoss)(trainer::InferOptTrainer; kwargs...)
    return sum(trainer.flux_loss(t...) for t in zip(get_data_test(trainer)...))
end

## ---

struct ErrorFunctionTrain <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

ErrorFunctionTrain(name="Train hamming distance") = ErrorFunctionTrain(name, Float64[])

function (m::ErrorFunctionTrain)(trainer::InferOptTrainer; Y_train_pred, kwargs...)
    train_error = mean(
        hamming_distance(y, y_pred) for (y, y_pred) in zip(trainer.dataset.Y_train, Y_train_pred)
    )
    return train_error
end

## ---

struct ErrorFunctionTest <: AbstractScalarMetric
    name::String
    history::Vector{Float64}
end

ErrorFunctionTest(name="Test hamming distance") = ErrorFunctionTest(name, Float64[])

function (m::ErrorFunctionTest)(trainer::InferOptTrainer; Y_test_pred, kwargs...)
    train_error = mean(
        hamming_distance(y, y_pred) for (y, y_pred) in zip(trainer.dataset.Y_train, Y_test_pred)
    )
    return train_error
end

## ---

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
