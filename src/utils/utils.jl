"""
$TYPEDEF

Callable struct that fixes the keyword arguments of `f` to `kwargs...`, and only accepts positional arguments.

# Fields
$TYPEDFIELDS
"""
struct FixKwargs{F,K}
    "function"
    f::F
    "fixed keyword arguments"
    kwargs::K
end

(fk::FixKwargs)(args...) = fk.f(args...; fk.kwargs...)

"""
$TYPEDEF

Callable struct that fixes the first argument of `f` to `x`, and the keyword arguments to `kwargs...`.

# Fields
$TYPEDFIELDS
"""
struct Fix1Kwargs{F,K,T} <: Function
    "function"
    f::F
    "fixed first argument"
    x::T
    "fixed keyword arguments"
    kwargs::K
end

(fk::Fix1Kwargs)(args...) = fk.f(fk.x, args...; fk.kwargs...)

"""
$TYPEDEF

Callable struct that fixes the first argument of `f` to `x`.
Compared to Base.Fix1, works on functions with more than two arguments.
"""
struct FixFirst{F,T}
    f::F
    x::T
end

(fk::FixFirst)(args...) = fk.f(fk.x, args...)
