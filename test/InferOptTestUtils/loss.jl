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
    l = loss(maximizer(encoder(x); instance=x), y; instance=x)
    return l
end

function (pl::PipelineLossImitationθ)(x, θ, y)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x); instance=x), θ; instance=x)
end

function (pl::PipelineLossImitationθy)(x, θ, y)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x); instance=x), θ, y; instance=x)
end

function (pl::PipelineLossImitationLoss)(x, θ, y)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x); instance=x), (; y_true=y, θ_true=θ); instance=x)
end
