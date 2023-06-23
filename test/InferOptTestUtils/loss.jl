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

function (pl::PipelineLossExperience)(
    x, θ, y; maximizer_kwargs=NamedTuple(), loss_kwargs=NamedTuple()
)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x); maximizer_kwargs...); instance=x, loss_kwargs...)
end

function (pl::PipelineLossImitation)(
    x, θ, y; maximizer_kwargs=NamedTuple(), loss_kwargs=NamedTuple()
)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x); maximizer_kwargs...), y; loss_kwargs...)
end

function (pl::PipelineLossImitationθ)(
    x, θ, y; maximizer_kwargs=NamedTuple(), loss_kwargs=NamedTuple()
)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x); maximizer_kwargs...), θ; loss_kwargs...)
end

function (pl::PipelineLossImitationθy)(
    x, θ, y; maximizer_kwargs=NamedTuple(), loss_kwargs=NamedTuple()
)
    (; encoder, loss, maximizer) = pl
    return loss(maximizer(encoder(x); maximizer_kwargs...), θ, y; loss_kwargs...)
end

function (pl::PipelineLossImitationLoss)(
    x, θ, y; maximizer_kwargs=NamedTuple(), loss_kwargs=NamedTuple()
)
    (; encoder, loss, maximizer) = pl
    return loss(
        maximizer(encoder(x); maximizer_kwargs...), (; y_true=y, θ_true=θ); loss_kwargs...
    )
end
