"""
    estimate_transitions([t, ] x, config::TransitionsSurrogatesConfig)

Estimate possible transitions for input timeseries `x` according to the configuration.
If `t` (the time vector of `x`), is not provided, it is assumed `t = eachindex(x)`.

Return the output as [`TransitionsResults`](@ref).
You can use this output also in [`transition_flags`](@ref).
"""
function estimate_transitions(x, config::TransitionsSurrogatesConfig)
    t = eachindex(x)
    return estimate_transitions(t, x, config)
end

function estimate_transitions(t::AbstractVector, x, config::TransitionsSurrogatesConfig)
    # initialize time vectors
    t_indicator = windowmap(config.whichtime, t; width = config.width_ind, stride = config.stride_ind)
    t_change = windowmap(config.whichtime, t_indicator; width = config.width_cha, stride = config.stride_cha)
    len_ind = length(t_indicator)
    len_change = length(t_change)
    # initialize array containers
    X = eltype(x)
    n_ind = length(config.indicators)
    x_indicator = zeros(X, len_ind, n_ind)
    x_change = zeros(X, len_change, n_ind) # same size, no matter how many change metrics

    pvalues = zeros(X, len_change, n_ind)
    if config.tail == :both
        pval_right = zeros(X, len_change)
        pval_left = zeros(X, len_change)
    end
    indicator_dummy = zeros(X, len_ind)
    change_dummy = zeros(X, len_change)
    # TODO: GPU usage dispatch on Random Fourier
    seeds = rand(config.rng, 1:typemax(Int), Threads.nthreads())
    sgens = [surrogenerator(x, config.surrogate, Random.Xoshiro(seed)) for seed in seeds]
    # sgen = surrogenerator(x, config.surrogate, config.rng)

    # TODO: Impose function barrier here!
    # TODO: the way we obtain the change metric is type unstable!
    # We probably need a function call for _each_ change metric!!!
    # This also satisfies the function barrier!!!
    # This also helps with doing optimizations for GPU!

    # Actual computations
    if length(config.change_metrics) == n_ind
        one2one = true
    elseif length(config.change_metrics) == 1
        one2one = false
    end
    for i in 1:n_ind # loop over indicators / change metrics
        indicator::Function = config.indicators[i]
        i_metric = one2one ? i : 1
        chametric::Function = config.change_metrics[i_metric]
        # indicator timeseries
        z = view(x_indicator, :, i)
        # change metric timeseries
        c = view(x_change, :, i)
        # p values for current i
        pval = view(pvalues, :, i)

        indicator_metric_surrogates_loop!(
            indicator, chametric, z, c, pval, x, config, sgens,
            indicator_dummy, change_dummy, pval_right, pval_left
        )

    end
    pvalues ./= config.n_surrogates

    # put everything together in the output type
    return TransitionsResults(
        t, x,
        config.indicators, t_indicator, x_indicator,
        config.change_metrics, t_change, x_change, pvalues,
        config.surrogate,
    )
end


function indicator_metric_surrogates_loop!(
        indicator, chametric, z, c, pval, x, config, sgens,
        indicator_dummy, change_dummy, pval_right, pval_left
    )
    windowmap!(indicator, z, x;
        width = config.width_ind, stride = config.stride_ind)
    windowmap!(chametric, c, z;
        width = config.width_cha, stride = config.stride_cha)
    # surrogates
    # TODO: parallelize over surrogates via threads here
    Threads.@threads for j in 1:config.n_surrogates
        sgen = sgens[Threads.threadid()]
        s = sgen()
        windowmap!(indicator, indicator_dummy, s;
            width = config.width_ind, stride = config.stride_ind)
        windowmap!(chametric, change_dummy, indicator_dummy;
            width = config.width_cha, stride = config.stride_cha
        )
        # This should be replaced by a simple call of pvalue() in future.
        # However, the use of pvalue() is less trivial for an incremental
        # computation over the surrogates.
        if config.tail == :right
            pval .+= c .< change_dummy
        elseif config.tail == :left
            pval .+= c .> change_dummy
        elseif config.tail == :both
            pval_right .+= c .< change_dummy
            pval_left .+= c .> change_dummy
        end
    end
    if config.tail == :both
        pval .= 2min.(pval_right, pval_left)
        pval_left .= pval_right .= 0
    end
end


"""
    transition_flags(results::TransitionsResults, p_threshold::Real) → flags

Return `flags::Matrix{Bool}`, which is an `n × (c+1)` sized matrix.
Each column `flags[:, c]` is the Boolean flags that correspond
to a p-value below the threshold `p` for each change metric timeseries.
There are `c` change metric timeseries in total, but the `c+1`-th column
of `flags` is simply the Boolean addition of all previous columns
(i.e., time points where _all_ indicators pass the significance).

To obtain the time windows where the `c`-th change metric is significant,
simply do:
```julia
timepoints = results.t_change[flags[:, c]]
```
"""
function transition_flags(results::TransitionsResults, p::Real)
    if p >= 1 || p <= 0
        error("Threshold must be a value between 0 and 1.")
    end
    thresholded_p = results.pvalues .< p
    lastcol = reduce(&, thresholded_p; dims = 2)
    flags = hcat(thresholded_p, lastcol)
    return flags
end