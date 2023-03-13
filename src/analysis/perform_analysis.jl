
"""
    analyze_indicators(t, x, indicators, evolution_metrics, p) → res::IndicatorEvolutionResults

Return the `indicators` and their `evolution_metrics` as [`IndicatorEvolutionResults`](@ref IndicatorEvolutionResults)
for a timeseries `t`, `x` and its surrogates.
If `t` is not provided, it is simply assumed to be `1:length(x)`.
This meta-analysis is performed based on `p::SignificanceHyperParams`.
"""
function analyze_indicators(
    t::AbstractVector{T},
    x::AbstractVector{T},
    indicators::Vector{Function},
    evolution_metrics::Vector{Function},
    p::SignificanceHyperParams,
    # TODO:
    midpoint_function,
) where {T<:AbstractFloat}

    n_ind = length(indicators)
    indicator_length = get_windowmapview_length(x,
        p.wv_indicator_width, p.wv_indicator_stride)
    evolution_length = get_windowmapview_length(1:indicator_length,
        p.wv_evolution_width, p.wv_evolution_stride)

    X_indicator = fill(T(0), 1, indicator_length, n_ind)
    X_evolution = fill(T(0), 1, evolution_length, n_ind)
    t_indicator = fill(T(0), indicator_length)
    t_evolution = fill(T(0), evolution_length)

    for i in 1:n_ind
        if i == 1       # t_indicator and t_evolution only need to be computed once.
            t1, x_indicator, t2, x_evolution = indicator_evolution(
                t, x, indicators[i], evolution_metrics[i], p)
            copy!(t_indicator, t1)
            copy!(t_evolution, t2)
        else
            x_indicator, x_evolution = indicator_evolution(
                x, indicators[i], evolution_metrics[i], p)
        end
        X_indicator[1, :, i] .= x_indicator
        X_evolution[1, :, i] .= x_evolution
    end

    sgen = surrogenerator(x, p.surrogate_method, p.rng)
    S_evolution = fill(T(0.0), p.n_surrogates, evolution_length, n_ind)
    for j in 1:p.n_surrogates
        for i in 1:n_ind
            s = sgen()
            s_indicator, s_evolution = indicator_evolution(
                s, indicators[i], evolution_metrics[i], p)
            S_evolution[j, :, i] .= s_evolution
        end
    end
    return IndicatorEvolutionResults(
        t_indicator, X_indicator, t_evolution, X_evolution, S_evolution,
    )
end

# allow for a single indicator and a single evolution metric.
function analyze_indicators(
    t::AbstractVector{T},
    x::AbstractVector{T},
    indicators::Function,
    evolution_metrics::Function,
    p::SignificanceHyperParams,
) where {T<:AbstractFloat}
    return analyze_indicators(t, x, Function[indicators], Function[evolution_metrics], p)
end

# allow for multiple indicators and a single evolution metric.
function analyze_indicators(
    t::AbstractVector{T},
    x::AbstractVector{T},
    indicators::Vector{Function},
    evolution_metrics::Function,
    p::SignificanceHyperParams,
) where {T<:AbstractFloat}
    em = repeat(Function[evolution_metrics], outer = length(indicators))
    return analyze_indicators(t, x, indicators, em, p)
end

# allow the user to not provide any time vector.
function analyze_indicators(
    x::AbstractVector{T},
    indicators::VFi,
    evolution_metrics::VFe,
    p::SignificanceHyperParams,
) where {
    T<:AbstractFloat,
    F<:Function,
    VFi<:Union{F, Vector{F}},
    VFe<:Union{F, Vector{F}},
}
    t = 1:length(x)
    return analyze_indicators(t, x, indicators, evolution_metrics, p)
end

"""
    indicator_evolution(x, indicator, evolution, p)
    indicator_evolution(t, x, indicator, evolution, p)

Based on `p::SignificanceHyperParams`, compute an `indicator` and its `evolution_metric` for
a timeseries `t`, `x`.
If `t` is not provided, it is simply assumed to be `1:length(x)`.
"""
function indicator_evolution(
    t::AbstractVector{T},
    x::Vector{T},
    indicator::Function,
    evolution_metric::Function,
    p::SignificanceHyperParams,
) where {T<:AbstractFloat}
    t_indicator, x_indicator = windowmap(t, x, indicator,
        p.wv_indicator_width, p.wv_indicator_stride)
    t_evolution, x_evolution = windowmap(t_indicator, x_indicator, evolution_metric,
        p.wv_evolution_width, p.wv_evolution_stride)

    return t_indicator, x_indicator, t_evolution, x_evolution
end

function indicator_evolution(
    x::Vector{T},
    indicator::Function,
    evolution_metric::Function,
    p::SignificanceHyperParams,
) where {T<:AbstractFloat}
    x_indicator = windowmap(x, indicator, p.wv_indicator_width, p.wv_indicator_stride)
    x_evolution = windowmap(x_indicator, evolution_metric,
        p.wv_evolution_width, p.wv_evolution_stride)

    return x_indicator, x_evolution
end
