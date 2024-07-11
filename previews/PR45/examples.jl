# # Examples for TransitionsInTimeseries.jl

# ## Permutation entropy for dynamic regime changes

# Permutation entropy is used frequently to detect a transition between
# one dynamic regime to another. It is useful when the mean and std. of the
# timeseries values are very similar between the two regimes, which would mean that
# common distribution-based indicators, or common critical-slowing-down based indicators,
# would fail.

# This example will also explore different ways to test for significance that
# are arguably better suitable in such an application than the [Tutorial](@ref)'s default
# of significance via random Fourier surrogates.

# ### Logistic map timeseries

# A simple example of this is transitions from periodic to weakly chaotic to chaotic
# motion in the logistic map. First, let's generate a timeseries of the logistic map

using DynamicalSystemsBase
using CairoMakie

logistic_rule(u, p, t) = @inbounds SVector(p[1]*u[1]*(1 - u[1]))
ds = DeterministicIteratedMap(logistic_rule, [0.5], [1.0])

r1 = 3.83
r2 = 3.86
N = 2000
rs = range(r1, r2; length = N)
x = zeros(N)
for (i, r) in enumerate(rs)
    set_parameter!(ds, 1, r)
    step!(ds)
    x[i] = current_state(ds)[1]
end

# Plot it, using as time the parameter value (they coincide)
fig, ax = lines(rs, x; linewidth = 0.5)
ax.xlabel = "r (time)"
ax.ylabel = "x"
fig

# In this example there is a rather obvious transition to strongly
# chaotic motion at r ≈ 3.855. However, there is also a subtle transition
# to weak chaos at r ≈ 3.847. This transition is barely visible in the
# timeseries, and in fact many of the timeseries statistical properties remain identical.

# ### Using a simpler change metric

# Now, let's compute and various indicators and their changes,
# focusing on the fourth indicator, the permutation entropy. We use
# order 4 here, because we know that to detect changes in a period `m` we would need
# an order ≥ `m+1` permutation entropy.

using TransitionsInTimeseries
indicators = (var, ar1_whitenoise, permutation_entropy(m = 4))
indistrings = ("var", "ar1", "pe")

# In this example there is no critical slowing down;
# instead, there is a sharp transition between periodic and chaotic motion.
# Hence, we shouldn't be using any trend-based change metrics.
# Instead, we will use the most basic change metric, [`difference_of_means`](@ref).
# With this metric it also makes most sense to use as stride half the window width
metric = difference_of_means

width_ind = N÷100
width_cha = 20
stride_cha = 10

config = WindowedIndicatorConfig(indicators, metric;
    width_ind, width_cha, stride_cha,
)

results = estimate_indicator_changes(config, x, rs)

# Let's now plot the change metrics of the indicators

function plot_change_metrics()
    fig, ax = lines(rs, x; axis = (ylabel = "input",), figure = (resolution = (600, 600),))
    hidexdecorations!(ax; grid = false)
    ## plot all change metrics
    for (i, c) in enumerate(eachcol(results.x_change))
        ax, = scatterlines(fig[i+1, 1], results.t_change, c;
            axis = (ylabel = indistrings[i],), label = "input"
        )
        if i < 3
            hidexdecorations!(ax; grid = false)
        else
            ax.xlabel = "r (time)"
        end
    end
    return fig
end

fig = plot_change_metrics()

# We already see the interesting results we expect: the permutation entropy shows a
# striking change as we go from periodic to weakly chaotic motion at r ≈ 3.847.
# (Remember: the plotted quantity is how much the indicator changes within a time window.
# High values mean large changes.)

# Due to its construction, permutation entropy will have a spike for periodic data at the
# start of the timeseries, so we can safely ignore the spike at r ≈ 3.83.

# ### Significance via random Fourier surrogates

# One way to test for significance would be via the standard way as in the [Tutorial](@ref),
# utilizing surrogate timeseries and [`SurrogatesSignificance`](@ref).

# Let's do it here for an example, but, we have to be **careful**.
# It is crucial that for permutation entropy we use `:right` as the `tail`,
# because it is expected that the surrogates will have higher
# differences in the permutation entropy timeseries (because, if there is no
# dynamical change, the permutation entropy will stay the same, while in the surrogates
# there are always random fluctuations!

surromethod = RandomFourier()

## Define a function because we will re-use later
function overplot_surrogate_significance!(fig, surromethod, color = "black")

    signif = SurrogatesSignificance(;
        n = 1000, tail = [:both, :both, :right], surromethod
    )
    flags = significant_transitions(results, signif)

    ## and also plot the flags with same color
    for (i, indicator) in enumerate(indicators)
        ## To make things visually clear, we will also plot some example surrogate
        ## timeseries for each indicator and change metric pair
        for _ in 1:10
            s = TimeseriesSurrogates.surrogate(x, surromethod)
            p = windowmap(indicator, s; width = width_ind)
            q = windowmap(metric, p; width = width_cha, stride = stride_cha)
            lines!(fig[i+1, 1], results.t_change, q;  color = (color, 0.2), linewidth = 1)
        end
        ## Plot the flags as vertical dashed lines
        vlines!(fig[i+1, 1], results.t_change[flags[:, i]];
            color = color, linestyle = :dash, linewidth = 3
        )
    end
    ## add a title to the figure with how we estimate significance
    content(fig[1, 1]).title = "surrogates: "*string(nameof(typeof(surromethod)))
end

surromethod = RandomFourier()
overplot_surrogate_significance!(fig, surromethod)

fig

# ### More appropriate surrogates
# %% #src
# Using random Fourier surrogates does not make much sense in our application.
# Those surrogates perserve the power spectrum of the timeseries, but the power spectrum
# is a property integrated over the whole timeseries. It doesn't contain any information
# regarding a sharp transition at some point in the timeseries.
# A much better alternative is to use block-shuffled surrogates, which
# preserve the short term local temporal correlation in the timeseries and hence
# also preserve local short term sharp changes in the dynamic behavior.

surromethod = BlockShuffle(15)
fig = plot_change_metrics()
overplot_surrogate_significance!(fig, surromethod, "red")
fig

# The results are better for the variance and AR1 indicators.
# For the permutation entropy the results do not change because it already is an
# exceptionally well
# suited indicator for this application scenario. But in other cases where things
# are not as clear, or data are contaminated with noise, or we have shorter data,
# choosing a more suitable
# surrogate generator may make the difference between a false positive or not.

# ### Simpler Significance
# %% #src
# Arguably, exactly because we are using the [`difference_of_means`](@ref) as a
# change metric, we may want to be much less strict with our tests for significance.
# Instead of using [`SurrogatesSignificance`](@ref) we may use the simpler and much faster
# [`QuantileSignificance`](@ref), which simply claims significant time points
# whenever a change metric exceeds some pre-defined quantile of its timeseries.

fig = plot_change_metrics()
flags = significant_transitions(results, QuantileSignificance())

## Plot the flags
for (i, indicator) in enumerate(indicators)
    vlines!(fig[i+1, 1], results.t_change[flags[:, i]];
        color = Cycled(3), linestyle = :dash, linewidth = 3
    )
end
content(fig[1, 1]).title = "significance from quantile"
fig
