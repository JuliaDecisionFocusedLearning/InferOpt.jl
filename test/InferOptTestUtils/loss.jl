abstract type PipelineLoss end

struct PipelineLossExperience{E,M,L,MK,LK} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
    maximizer_kwargs::MK
    loss_kwargs::LK
end

struct PipelineLossImitation{E,M,L,MK,LK} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
    maximizer_kwargs::MK
    loss_kwargs::LK
end

struct PipelineLossImitationθ{E,M,L,MK,LK} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
    maximizer_kwargs::MK
    loss_kwargs::LK
end

struct PipelineLossImitationθy{E,M,L,MK,LK} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
    maximizer_kwargs::MK
    loss_kwargs::LK
end

struct PipelineLossImitationLoss{E,M,L,MK,LK} <: PipelineLoss
    encoder::E
    maximizer::M
    loss::L
    maximizer_kwargs::MK
    loss_kwargs::LK
end

function (pl::PipelineLossExperience)(x, θ, y)
    (; encoder, loss, maximizer, maximizer_kwargs, loss_kwargs) = pl
    return loss(maximizer(encoder(x); maximizer_kwargs...); instance=x, loss_kwargs...)
end

function (pl::PipelineLossImitation)(x, θ, y)
    (; encoder, loss, maximizer, maximizer_kwargs, loss_kwargs) = pl
    return loss(maximizer(encoder(x); maximizer_kwargs...), y; loss_kwargs...)
end

function (pl::PipelineLossImitationθ)(x, θ, y)
    (; encoder, loss, maximizer, maximizer_kwargs, loss_kwargs) = pl
    return loss(maximizer(encoder(x); maximizer_kwargs...), θ; loss_kwargs...)
end

function (pl::PipelineLossImitationθy)(x, θ, y)
    (; encoder, loss, maximizer, maximizer_kwargs, loss_kwargs) = pl
    return loss(maximizer(encoder(x); maximizer_kwargs...), θ, y; loss_kwargs...)
end

function (pl::PipelineLossImitationLoss)(x, θ, y)
    (; encoder, loss, maximizer, maximizer_kwargs, loss_kwargs) = pl
    return loss(
        maximizer(encoder(x); maximizer_kwargs...), (; y_true=y, θ_true=θ); loss_kwargs...
    )
end
