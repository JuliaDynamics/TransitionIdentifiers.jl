using TransitionIndicators, CairoMakie

t, x_linear, x_nlinear = load_linear_vs_doublewell()
fig, ax = lines(t, x_nlinear)
lines!(ax, t, x_linear)
ax.title = "raw data"
fig

#
x_l_fluct = diff(x_linear)
x_nl_fluct = diff(x_nlinear)
tfluct = t[2:end]

fig, ax = lines(tfluct, x_l_fluct)
lines!(ax, tfluct, x_nl_fluct .+ 0.05)
ax.title = "input timeseries"
fig

#
indicator = ar1_whitenoise
indicator_window = (width = 400, stride = 1)

# By mapping `last::Function` over a windowviewer of the time vector, we obtain the last time step of each window.
# This therefore only uses information from `k-width` to `k` at time step `k`.
# Alternatives: `first::Function`, `midpoint:::Function`.
t_indicator = windowmap(last, tfluct; indicator_window...)
indicator_l = windowmap(indicator, x_l_fluct; indicator_window...)
indicator_nl = windowmap(indicator, x_nl_fluct; indicator_window...)

fig, ax = lines(t_indicator, indicator_l)
lines!(ax, t_indicator, indicator_nl)
ax.title = "indicator timeseries"
fig

#
change_window = (width = 30, stride = 1)
ridgereg = RidgeRegressionSlope(lambda = 0.0)
precompridgereg = precompute(ridgereg, t[1:change_window.width])

t_change = windowmap(last, t_indicator; change_window...)
change_l = windowmap(precompridgereg, indicator_l; change_window...)
change_nl = windowmap(precompridgereg, indicator_nl; change_window...)

fig, ax = lines(t_change, change_l)
lines!(ax, t_change, change_nl)
ax.title = "change metric timeseries"
fig


#

# Generate Fourier random-phase surrogates
using Random: Xoshiro
s = surrogate(x_nl_fluct, RandomFourier(), Xoshiro(123))
fig, ax = lines(tfluct, x_nl_fluct; color = Cycled(2))
lines!(ax, tfluct, s .- 0.05; color = Cycled(3))
ax.title = "real signal vs. surrogate(s)"

# compute and plot indicator and change metric
indicator_s = windowmap(indicator, s; indicator_window...)
change_s = windowmap(precompridgereg, indicator_s; change_window...)

ax, = lines(fig[1,2], t_change, change_nl; color = Cycled(2), label = "nonlin")
lines!(ax, t_change, change_s; color = Cycled(3), label = "surrogate")
axislegend()
ax.title = "change metric"

fig
#


n_surrogates = 1_000
fig = Figure()
axl = Axis(fig[1,1]; title = "linear")
axnl = Axis(fig[1,2]; title = "nonlinear")
axsigl = Axis(fig[2,1])
axsignl = Axis(fig[2,2])

for (j, ax, axsig, x) in zip(1:2, (axl, axnl), (axsigl, axsignl), (x_l_fluct, x_nl_fluct))

    orig_change = j == 1 ? change_l : change_nl
    sgen = surrogenerator(x, RandomFourier(), Xoshiro(123))
    pval = zeros(length(change_s))

    # Collect all surrogate change metrics
    for i in 1:n_surrogates
        s = sgen()
        indicator_s = windowmap(indicator, s; indicator_window...)
        change_s = windowmap(precompridgereg, indicator_s; change_window...)
        pval += orig_change .< change_s
    end

    pval ./= n_surrogates
    lines!(ax, t_change, orig_change; color = Cycled(j))
    lines!(axsig, t_change, pval; color = Cycled(j+2))
end

fig

# %%
# Tutorial short

using TransitionIndicators, CairoMakie

t, x_linear, x_nlinear = load_linear_vs_doublewell()

x_nl_fluct = diff(x_nlinear)
tfluct = t[2:end]

fig, ax = lines(tfluct, x_nl_fluct)
ax.title = "input timeseries"
fig

# These indicators are suitable for Critical Slowing Down
indicators = [var, ar1_whitenoise]
indconfig = IndicatorsConfig(tfluct, last, indicators; width = 400)


# use the ridge regression slope for both indicators
change_metrics = [RidgeRegressionSlope()]
sigconfig = SignificanceConfig(indconfig, last, change_metrics;
    width = 30, n_surrogates = 1000)

# perform the full analysis
results = indicators_analysis(tfluct, x_nl_fluct, indconfig, sigconfig)

# Plot the original timeseries vs. p-value time series
fig, ax = lines(tfluct, x_nl_fluct; color = Cycled(2), label = "input")
axpval, = lines(fig[2,1], sigconfig.t_change, results.pval[:, 1]; color = Cycled(3), label = "p-value of var")
lines!(axpval, sigconfig.t_change, results.pval[:, 2]; color = Cycled(4), label = "p-value of ar1")
xlims!(ax, (0, 50))
xlims!(axpval, (0, 50))
axislegend(axpval)
fig


flags_indicators, flags_andicators = transition_flags(results, 0.05)
vlines!(ax, flags_andicators; label = "flags", color = Cycled(3), linestyle = :dash)
axislegend(ax)
fig