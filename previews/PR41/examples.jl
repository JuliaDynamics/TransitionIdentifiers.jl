# # Examples for TransitionsInTimeseries.jl

# ## Permutation entropy

# Permutation entropy is used frequently to detect a transition between
# one dynamic regime to another. It is useful when the mean and std. of the
# timeseries values are very similar between the two regimes, which would mean that
# common distribution-based indicators, or common critical-slowing-down based indicators,
# would fail. A simple example of this is transitions from periodic to periodic to chaotic
# motion in the logistic map. First, let's generate a timeseries of the logistic map

using TransitionsInTimeseries
using DynamicalSystemsBase
using CairoMakie

logistic_rule(u, p, t) = @inbounds SVector(p[1]*u[1]*(1 - u[1]))
ds = DeterministicIteratedMap(logistic_rule, [0.5], [1.0])

r1 = 3.83
# r1 = 3.8
r2 = 3.86
# r2 = 3.9
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
# chaotic motion at r ≈ 3.857. However, there is also a subtle transition
# to weak chaos at r ≈ 3.849. This transition is barely visible in the
# timeseries.

# Now, let's compute and plot various indicators
indicators = [mean, var, ar1_whitenoise, PermutationEntropy(m = 4)]
indistrings = ["mean", "var", "ar1", "pe"]
# In this example there is no critical slowing down;
# instead, there is a sharp transition between period-3 and chaotic motion.
# Hence, we shouldn't be using any trend-based change metrics.
# Instead, we will use the most basic change metric, [`difference_of_means`](@ref).
metric = difference_of_means

# And we perform the standard package analysis as in the [Tutorial](@ref),
# but with a crucial difference: the type of surrogates to generate _must_ be
# different. Fourier TODO:
surrogate = RandomFourier()

config = TransitionsSurrogatesConfig(indicators, metric, surrogate;
    width_ind = N÷100, stride_ind = 5, width_cha = 2, n_surrogates = 100
)

results = estimate_transitions(rs, x, config)

flags = transition_flags(results, 0.05)

# And we visualize the timeseries and some
# %% #src
fig, ax = lines(rs, x; axis = (ylabel = "input",))
hidexdecorations!(ax; grid = false)
## plot all change metrics
for (i, c) in enumerate(eachcol(results.x_change))
    # plot change metric
    ax, = scatterlines(fig[i+1, 1], results.t_change, c, axis = (ylabel = indistrings[i],))
    #
    i < 4 && hidexdecorations!(ax; grid = false)
end
fig