abstract type PipelineLoss end

struct PipelineLossExperience <: PipelineLoss end
struct PipelineLossImitation <: PipelineLoss end
struct PipelineLossImitationθ <: PipelineLoss end
struct PipelineLossImitationθy <: PipelineLoss end
struct PipelineLossImitationLoss <: PipelineLoss end

get_loss(::PipelineLossExperience, loss, res, x, θ, y) = loss(res; instance=x)
get_loss(::PipelineLossImitation, loss, res, x, θ, y) = loss(res, y; instance=x)
get_loss(::PipelineLossImitationθ, loss, res, x, θ, y) = loss(res, θ; instance=x)
get_loss(::PipelineLossImitationθy, loss, res, x, θ, y) = loss(res, θ, y; instance=x)
function get_loss(::PipelineLossImitationLoss, loss, res, x, θ, y)
    return loss(res, (; y_true=y, θ_true=θ); instance=x)
end
