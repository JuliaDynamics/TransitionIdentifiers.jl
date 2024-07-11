using TransitionIndicators, CairoMakie

t, x_linear, x_nlinear = load_linear_vs_doublewell()
fig, ax = lines(t, x_nlinear)
lines!(ax, t, x_linear)
ax.title = "raw data"
fig

#
x_nl_fluct = diff(x_nlinear)
x_l_fluct = diff(x_linear)
tfluct = t[2:end]

fig, ax = lines(tfluct, x_nl_fluct)
ax.title = "input timeseries"
fig

#
indicator = ar1_whitenoise
indicator_window = (width = 400, stride = 1)

t_indicator = windowmap(midpoint, tfluct; indicator_window...)
indicator_nl = windowmap(indicator, x_nl_fluct; indicator_window...)
indicator_l = windowmap(indicator, x_l_fluct; indicator_window...)

fig, ax = lines(t_indicator, indicator_nl)
ax.title = "indicator timeseries"
fig

#
change_window = (width = 20, stride = 1)
ridgereg = RidgeRegression(t_indicator, change_window.width)

t_change = windowmap(midpoint, t_indicator; change_window...)
change_l = windowmap(ridgereg, indicator_l; change_window...)
change_nl = windowmap(ridgereg, indicator_nl; change_window...)

fig, ax = lines(t_change, change_nl)
# lines!(ax, t_change, change_l)
ax.title = "change metric timeseries"
fig


#

## Generate Fourier random-phase surrogates
using Random: Xoshiro
s = surrogate(x_nl_fluct, RandomFourier(), Xoshiro(123))
fig, ax = lines(tfluct, x_nl_fluct)
lines!(ax, tfluct, s .- 0.05; color = Cycled(3))
ax.title = "real signal vs. surrogate(s)"

# compute and plot indicator and change metric
indicator_s = windowmap(indicator, s; indicator_window...)
change_s = windowmap(ridgereg, indicator_s; change_window...)
fig, ax = lines(t_indicator, indicator_nl)
lines!(ax, t_indicator, indicator_s; color = Cycled(3))
ax.ylabel = "indicator"
ax, = lines(fig[1,2], t_change, change_nl)
lines!(ax, t_change, change_s; color = Cycled(3))
ax.ylabel = "change metric"
Label(fig[0, :], "real signal vs. surrogate(s)";
    tellheight = true, tellwidth = false, valign = :bottom, font = "TeX Gyre Heros Bold"
)
fig




lines!(ax, tfluct, s .- 0.05; color = Cycled(3))
ax.title = "real signal vs. surrogate(s)"



ax, = lines(fig[1,2], t_change, change_nl; label = "real")
lines!(ax, t_change, change_s; color = Cycled(3), label = "surrogate")
axislegend()
ax.title = "change metric"

fig

# %%

n_surrogates = 1_000
fig = Figure()
axnl = Axis(fig[1,1]; title = "nonlinear")
# axl = Axis(fig[1,2]; title = "linear")
axl = nothing

for (j, ax, x) in zip(1:2, (axnl, axl), (x_nl_fluct, x_l_fluct))

    sgen = surrogenerator(x, RandomFourier(), Xoshiro(123))
    change_s_distr =  zeros(n_surrogates, length(change_s))

    # Collect all surrogate change metrics
    for i in 1:n_surrogates
        s = sgen()
        indicator_s = windowmap(indicator, s; indicator_window...)
        change_s = windowmap(ridgereg, indicator_s; change_window...)
        change_s_distr[i, :] .= change_s
    end

    mu = vec(mean(change_s_distr, dims = 1))
    sigma = vec(std(change_s_distr, dims = 1))

    # Plot (real signal) change metric and various confidence intervals
    orig_change = j == 1 ? change_nl : change_l
    lines!(ax, t_change, orig_change; color = Cycled(j))
    # band!(ax, t_change, mu .- 2 .* sigma, mu .+ 2 .* sigma, color = (:red, 0.2) )
    band!(ax, t_change, mu .- 3 .* sigma, mu .+ 3 .* sigma, color = (Main.COLORS[3], 0.2),
    label = "±3σ")
    axislegend(ax)
    j == 1 && break
end

fig

# %%

# these indicators are suitable for Critical Slowing Down
indicators = [var, ar1_whitenoise]
ind_conf = IndicatorsConfig(indicators; width = 400)

# use spearman correlation for both indicators
change_metric = spearman
sig_conf = SignificanceConfig(change_metric; width = 20, n_surrogates = 1000)

# perform the full analysis
result = indicators_analysis(x_nl_fluct, ind_conf, sig_conf)

# And lastly, obtain some flags for when there is a significant indicator change
sig = indicators_significance(result, 0.99)

# Plot the original timeseries
fig, ax = lines(tfluct, x_nl_fluct; label = "input")
# Scatter the significance of each indicator
for i in size(sig, 2)
    signif_idxs = findall(sig[:, i])
    isempty(signif_idxs) && continue
    # get timepoints in real time
    tflags = tfluct[result.t_change[signif_idxs]]
    vlines!(ax, tflags; label = "flags for $(indicators[i])", color = Cycled(3))
end
axislegend()
fig



# Tutorial short

using TransitionIndicators, CairoMakie

t, x_linear, x_nlinear = load_linear_vs_doublewell()

fig, ax = lines(t, x_nlinear;z)
ax.title = "original timeseries"
fig

input = diff(x_nlinear)
t = t[2:end]

fig, ax = lines(tfluct, input; color = Cycled(2))
ax.title = "input timeseries"
fig



# these indicators are suitable for Critical Slowing Down
indicators = [var, ar1_whitenoise]
ind_conf = IndicatorsConfig(indicators;
    width = 400
)
# use spearman correlation for both indicators
change_metric = spearman
sig_conf = SignificanceConfig(change_metric;
    width = 20, n_surrogates = 10_000,
    surrogate_method = RandomFourier(),
)

# perform the full analysis
result = indicators_analysis(input, ind_conf, sig_conf)

# And lastly, obtain some flags for when there
# is a significant indicator change
sig = indicators_significance(result, 0.99)

sig

# Plot the original timeseries
fig, ax = lines(tfluct, x_nl_fluct; color = Cycled(2), label = "input")
# Scatter the significance of each indicator
for i in size(sig, 2)
    signif_idxs = findall(sig[:, i])
    isempty(signif_idxs) && continue
    # get timepoints in real time
    tflags = tfluct[result.t_change[signif_idxs]]
    vlines!(ax, tflags; label = "indicator $(indicators[i])", color = Cycled(2+i))
end
axislegend()
fig


