"""
    DEFAULT_FRANK_WOLFE_KWARGS

Default configuration for the Frank-Wolfe wrapper.

# Parameters
- `away_steps`
- `epsilon`
- `lazy`
- `line_search`
- `max_iteration`
- `timeout`
- `verbose`
"""
const DEFAULT_FRANK_WOLFE_KWARGS = (
    away_steps=true,
    epsilon=1e-3,
    lazy=true,
    line_search=Adaptive(),
    max_iteration=100,
    timeout=1.0,
    verbose=false,
)

## Wrapper for linear maximizers to use them within Frank-Wolfe

struct LMOWrapper{F,K} <: LinearMinimizationOracle
    maximizer::F
    maximizer_kwargs::K
end

LMOWrapper(maximizer) = LMOWrapper(maximizer, (;))

function FrankWolfe.compute_extreme_point(lmo_wrapper::LMOWrapper, direction; kwargs...)
    (; maximizer, maximizer_kwargs) = lmo_wrapper
    v = maximizer(-direction; maximizer_kwargs...)
    return v
end
