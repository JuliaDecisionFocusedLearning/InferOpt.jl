abstract type PipelineLoss end

struct PipelineLossExperience{E,M,L} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
end

struct PipelineLossImitation{E,M,L} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
end

struct PipelineLossImitationθ{E,M,L} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
end

struct PipelineLossImitationθy{E,M,L} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
end

struct PipelineLossImitationLoss{E,M,L} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
end

function (pl::PipelineLossExperience)(x, θ, y)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x)); instance=x)
end

function (pl::PipelineLossImitation)(x, θ, y)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x)), y)
end

function (pl::PipelineLossImitationθ)(x, θ, y)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x)), θ)
end

function (pl::PipelineLossImitationθy)(x, θ, y)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x)), θ, y)
end

function (pl::PipelineLossImitationLoss)(x, θ, y)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x)), (; y_true=y, θ_true=θ))
end
