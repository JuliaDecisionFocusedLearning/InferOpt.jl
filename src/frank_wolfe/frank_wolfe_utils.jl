"""
    DEFAULT_FRANK_WOLFE_KWARGS

Default configuration for the Frank-Wolfe wrapper.

# Parameters
- `away_steps=true`: activate away steps to avoid zig-zagging
- `epsilon=1e-4`: precision
- `lazy=true`: caching strategy
- `line_search=FrankWolfe.Adaptive()`: step size selection
- `max_iteration=10`: number of iterations
- `timeout=1.0`: maximum time in seconds
- `verbose=false`: console output
"""
const DEFAULT_FRANK_WOLFE_KWARGS = (
    away_steps=true,
    epsilon=1e-4,
    lazy=true,
    line_search=Adaptive(),
    max_iteration=10,
    timeout=1.0,
    verbose=false,
)

## Wrapper for linear maximizers to use them within Frank-Wolfe

"""
    LMOWrapper{F,K}

Wraps a linear maximizer as a `FrankWolfe.LinearMinimizationOracle`.

# Fields
- `maximizer::F`: black box linear maximizer
- `maximizer_kwargs::K`: keyword arguments passed to the maximizer whenever it is called
"""
struct LMOWrapper{F,K} <: LinearMinimizationOracle
    maximizer::F
    maximizer_kwargs::K
end

LMOWrapper(maximizer) = LMOWrapper(maximizer, (;))

"""
    FrankWolfe.compute_extreme_point(lmo_wrapper::LMOWrapper, direction)
"""
function FrankWolfe.compute_extreme_point(lmo_wrapper::LMOWrapper, direction; kwargs...)
    (; maximizer, maximizer_kwargs) = lmo_wrapper
    v = maximizer(-direction; maximizer_kwargs...)
    return v
end
