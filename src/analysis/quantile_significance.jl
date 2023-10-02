"""
    QuantileSignificance(; p = 0.95, tail = :right) <: TransitionsSignificance

A configuration struct for significance testing [`significant_transitions`](@ref).
When used with [`WindowedIndicatorResults`](@ref), significance is estimated
by comparing the value of each change metric with its `p`-quantile.
Values that exceed the `p`-quantile (if `tail = :right`)
or subseed the `1-p`-quantile (if `tail = :left`)
are deemed significant.
If `tail = :both` then either condition is checked.

`QuantileSignficance` guarantees that some values will be significant by
the very definition of what a quantile is.
See also [`SigmaSignificance`](@ref) that is similar but does not have this guarantee.
"""
Base.@kwdef struct QuantileSignificance{P<:Real}
    p::P = 0.95
    tail::Symbol = :right
end

using Statistics: quantile

function significant_transitions(res::WindowedIndicatorResults, signif::QuantileSignificance)
    flags = similar(res.x_change, Bool)
    for (i, x) in enumerate(eachcol(res.x_change))
        qmin, qmax = quantile(x, (1 - signif.p, signif.p))
        flag = view(flags, :, i)
        if signif.tail == :right
            @. flag = x > qmax
        elseif signif.tail == :left
            @. flag = x < qmin
        elseif signif.tail == :both
            @. flag = (x < qmin) | (x > qmax)
        else
            error("`tail` can be only `:left, :right, :both`. Got $(tail).")
        end
    end
    return flags
end

"""
    SigmaSignificance(; m = 3.0, tail = :right) <: TransitionsSignificance

A configuration struct for significance testing [`significant_transitions`](@ref).
When used with [`WindowedIndicatorResults`](@ref), significance is estimated
by comparing how many standard deviations (`σ`) the value exceeds the mean value (`μ`).
Values that exceed (if `tail = :right`) `μ + m*σ`, or subseed (if `tail = :left`) `μ - m*σ`
are deemed significant.
If `tail = :both` then either condition is checked.

`m` can also be a vector of values,
in which case a different value is used for each change metric.

See also [`QuantileSignificance`](@ref).
"""
Base.@kwdef struct SigmaSignificance{P}
    m::P = 0.95
    tail::Symbol = :right
end

using Statistics: std, mean

function significant_transitions(res::WindowedIndicatorResults, signif::SigmaSignificance)
    flags = similar(res.x_change, Bool)
    for (i, x) in enumerate(eachcol(res.x_change))
        μ = mean(x)
        σ = std(x; mean = μ)
        m = signif.m isa AbstractVector ? signif.m[i] : m
        flag = view(flags, :, i)
        if signif.tail == :right
            @. flag = x > μ + m*σ
        elseif signif.tail == :left
            @. flag = x < μ - m*σ
        elseif signif.tail == :both
            @. flag = (x < μ - m*σ) | (x > μ + m*σ)
        else
            error("`tail` can be only `:left, :right, :both`. Got $(tail).")
        end
    end
    return flags
end

