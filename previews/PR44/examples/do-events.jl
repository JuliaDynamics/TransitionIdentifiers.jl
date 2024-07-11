#=
# Dansgaard-Oescher events

The $\delta^{18}O$ time series of the North Greenland Ice Core Project ([NGRIP](https://en.wikipedia.org/wiki/North_Greenland_Ice_Core_Project)) are, to this date, the best proxy record for the Dansgaard-Oeschger events ([DO-events](https://en.wikipedia.org/wiki/Dansgaard%E2%80%93Oeschger_event)). DO-events are sudden warming episodes of the North Atlantic, reaching to 10 degrees of regional warming within 100 years. They happened quasi-periodically over the last glacial cycle due to transitions between strong and weak states of the Atlantic Meridional Overturning Circulation and might be therefore be the most prominent examples of abrupt transitions in the field of climate science. We here propose to hindcast these events by applying the theory of Critical Slowing Down (CSD) on the NGRIP data, which can be found [here](https://www.iceandclimate.nbi.ku.dk/data/) in its raw format. This analysis has already been done in [^Boers2018] and we here try to reproduce Figure 2.d-f.

## Preprocessing NGRIP

Over this example, it will appear that the convenience of TransitionsInTimeseries to leads the bulk of the code to be written for plotting and preprocessing. The latter consists in various steps $i = \lbrace 1,2,3 \rbrace$:
1. Load the data, reverse and offset it to have time vector = time before 2000 AD.
2. Filter non-unique points in time and sort the data.
3. Regrid the data from uneven to even sampling.

The time and $\delta^{18}O$ vectors resulting from step $i$ are respectively called $t_i$ and $x_i$. The final preprocessing step consists in obtaining a residual $r$, i.e. the fluctuations of the system around the attractor, which, within the CSD theory, is assumed to be tracked.
=#

using DelimitedFiles, Downloads, DSP, BSplineKit, Loess

function load_ngrip()
    tmp = Base.download("https://raw.githubusercontent.com/JuliaDynamics/JuliaDynamics/"*
        "master/timeseries/NGRIP.csv")
    data, labels = readdlm(tmp, header = true)
    return reverse(data[:, 1]) .- 2000, reverse(data[:, 2]) # (time, delta-18-0) vectors
end

uniqueidx(v) = unique(i -> v[i], eachindex(v))
function keep_unique(t, x)
    unique_idx = uniqueidx(t)
    return t[unique_idx], x[unique_idx]
end

function sort_timeseries!(t, x)
    p = sortperm(t)
    permute!(t, p)
    permute!(x, p)
    return nothing
end

function regrid2evensampling(t, x, dt)
    itp = BSplineKit.interpolate(t, x, BSplineOrder(4))
    tspan = (ceil(minimum(t)), floor(maximum(t)))
    t_even = collect(tspan[1]:dt:tspan[2])
    x_even = itp.(t_even)
    return t_even, x_even
end

function chebyshev_filter(t, x, fcutoff)
    ii = 10     # Chebyshev filtering requires to prune first points of time series.
    responsetype = Highpass(fcutoff, fs = 1/dt)
    designmethod = Chebyshev1(8, 0.05)
    r = filt(digitalfilter(responsetype, designmethod), x)
    xtrend = x - r
    return t[ii:end], x[ii:end], xtrend[ii:end], r[ii:end]
end

dt, fcutoff = 5.0, 0.95*0.01    # dt = 5 yr and cutoff ≃ 0.01 yr^-1 as in (Boers 2018)
t1, x1 = load_ngrip()
t2, x2 = keep_unique(t1, x1)
sort_timeseries!(t2, x2)
t3, x3 = regrid2evensampling(t2, x2, dt)
t, x, xtrend, r = chebyshev_filter(t3, x3, fcutoff)

#=
Let's now go to the last preprocessing steps and visualize our data in what will become our main figure. To keep the same approach as in [^Boers2018], we hand-code the transition times and highlight them by purple vertical lines:
=#

using CairoMakie

function loess_filter(t, x; span = 0.005)
    loessmodel = loess(t, x, span = span)
    xtrend = Loess.predict(loessmodel, t)
    r = x - xtrend
    return t, x, xtrend, r
end

function kyr_xticks(tticks_yr)
    tticks_kyr = ["$t" for t in Int.(tticks_yr ./ 1e3)]
    return (tticks_yr, tticks_kyr)
end

function plot_do(traw, xraw, tfilt, xfilt, t, r, t_transitions, xlims, xticks)
    fig = Figure(resolution = (1600, 1200), fontsize = 24)

    ## Original time series with transition marked by vertical lines
    ax1 = Axis(fig[1, 1], xlabel = L"Time (kyr) $\,$", ylabel = L"$\delta^{18}$O (permil)",
        xaxisposition = :top, xticks = xticks)
    lines!(ax1, traw, xraw, color = (:gray70, 0.5))
    lines!(ax1, tfilt, xfilt, color = :gray10, linewidth = 3)
    vlines!(ax1, t_transitions, color = Cycled(1), linewidth = 3)

    ## Residual time series
    ax2 = Axis(fig[2, 1], ylabel = L"Residual $\,$", xticks = xticks,
        xticksvisible = false, xticklabelsvisible = false)
    lines!(ax2, t, r, color = :gray50, linewidth = 1)

    ## Axes for variance and AC1 time series
    ax3 = Axis(fig[3, 1], ylabel = L"Variance $\,$", xticks = xticks,
        xticksvisible = false, xticklabelsvisible = false)
    ax4 = Axis(fig[4, 1], xlabel = L"Time (kyr) $\,$", ylabel = L"Lag-1 autocor. $\,$",
        xticks = xticks)

    axs = [ax1, ax2, ax3, ax4]
    [xlims!(ax, xlims) for ax in axs]
    ylims!(axs[1], (-48, -34))
    rowgap!(fig.layout, 10)
    return fig, axs
end

xlims = (-60e3, -10e3)
xticks = kyr_xticks(-60e3:5e3:5e3)
t_transitions = [-59.5e3, -58.2e3, -55.8e3, -54.3e3, -49.35e3, -46.9e3, -43.45e3,
                 -41.5e3, -40.2e3, -38.25e3, -35.5e3, -33.8e3, -32.55e3, -28.95e3,
                 -27.85e3, -23.5e3, -14.7e3, -11.7e3]
tloess, _, xloess, rloess = loess_filter(t3, x3)    # loess-filtered signal for visualization

fig, axs = plot_do(t3, x3, tloess, xloess, t, r, t_transitions, xlims, xticks)
fig

#=
## Hindcast on NGRIP data

As one can see, there is not much to see so far. Residuals are impossible to simply eye-ball and we therefore use TransitionsInTimeseries to study the evolution, measured by the ridge-regression slope, of the residual's variance and lag-1 autocorrelation (AC1) over time. In many examples of the literature, including [^Boers2018], the CSD analysis is performed over segments (sometimes only one) of the time series, such that a significance value is obtained for each segment. Dealing with segments can be easily done in TransitionsInTimeseries and is demonstrated here:
=#

using TransitionsInTimeseries, StatsBase
using Random: Xoshiro

ac1(x) = sum(autocor(x, [1]))                   # AC1 from StatsBase
indicators = (var, ac1)
change_metrics = RidgeRegressionSlope()
opts = [(color = Cycled(2), linewidth = 3),     # plotting options
        (color = :orange, linewidth = 3, linestyle = :dash)]

function perform_segment_analysis(t, r, indicators, change_metrics, itime,
    dt, n, t_transitions, margins)

    ## Initialize results over segment and indicator dimensions
    pvalues = fill(Inf, length(t_transitions), length(indicators))
    indicator_results = [fill(Inf, 3, 3) for t in t_transitions]

    for i in eachindex(t_transitions)
        ## Select the correct segment limits
        if i == 1
            t_start = first(t)
        else
            t_start = t_transitions[i-1] + margins[1]
        end
        t_end = t_transitions[i] - margins[2]
        i_start, i_end = argmin(abs.(t .- t_start)), argmin(abs.(t .- t_end))
        tseg, rseg = t[i_start:i_end], r[i_start:i_end]
        sigconfig = SurrogatesConfig(n = n, tail = :right, rng = Xoshiro(27))

        if last(tseg) - first(tseg) > itime + 20.0     # only analyze if segment long enough
            ## Compute the significance of the transition metrics for the chosen segment
            t_indicator = windowmap(last, tseg; width = Int(itime ÷ dt))
            segconfig = WindowedIndicatorConfig(indicators, change_metrics; whichtime = last,
                width_ind = Int(itime ÷ dt), width_cha = length(t_indicator))
            segresults = transition_metrics(segconfig, rseg, tseg)
            segsignif = estimate_significance(sigconfig, segresults)

            ## Populate the pre-allocated arrays with the results
            pvalues[i, :] .= segsignif.pvalues[1, :]
            indicator_results[i] = hcat(segresults.t_indicator, segresults.x_indicator)
        end
    end
    return pvalues, indicator_results
end

function plot_segment_analysis!(axs, pvalues, t_transitions, indicator_results, margins)
    for i in eachindex(indicator_results)   # loop over the segments
        ## Unpack indicator results
        tind = indicator_results[i][:, 1]
        ind = indicator_results[i][:, 2:end]

        for j in axes(pvalues, 2)           # loop over the indicators
            if !isinf(pvalues[i, j])        # only plot if enough data points for analysis
                ## Plot indicator time series and its linear regression
                lines!(axs[j+2], tind, ind[:, j], color = Cycled(1))
                m, p = ridgematrix(tind, 0.0) * ind[:, j]
                if pvalues[i, j] < 0.05
                    vlines!(axs[1], t_transitions[i] - margins[2]; opts[j]...)
                    vlines!(axs[j+2], t_transitions[i] - margins[2]; opts[j]...)
                    lines!(axs[j+2], tind, m .* tind .+ p, color = :gray10, linewidth = 3)
                else
                    lines!(axs[j+2], tind, m .* tind .+ p, color = :gray10, linewidth = 3,
                    linestyle = :dash)
                end
            end
        end
    end
    return nothing
end

margins = [200, 200]        # avoid the inclusion of transition data points in the segments
pvalues, indicator_results = perform_segment_analysis(t, r,
    indicators, change_metrics, 200.0, dt, 1_000, t_transitions, margins)
plot_segment_analysis!(axs, pvalues, t_transitions, indicator_results, margins)
fig

#=
In [^Boers2018], 13/16 and 7/16 true positives are respectively found for the variance and AC1, with 16 referring to the total number of transitions. Here we respectively find 10/16 true positives for the variance and 4/16 for AC1. The mismatch between [^Boers2018] and the results shown above clearly points out that packages like TransitionsInTimeseries are wishful for research to be reproducible.

Given that the NGRIP data is sparse, noisy, and presents 16 transitions in total, it appears that the method generates a reasonable rate of true positives, but...

## Limitations of hindcast

Finding transition indicators by defining segments, as done above, is arguably misleading when it comes to an operational prediction task. In fact, one does not check for false positives since the analysis is always performed upto time steps shortly before the transition. To make a brief, albeit incomplete check, one can stop the analysis much before the transition and thus check whether CSD indicators would forecast a transition although there is none ahead.
=#

early_margins = [200, 700]      # take "too early" end margin to check for false positives
fig, axs = plot_do(t3, x3, tloess, xloess, t, r, t_transitions, xlims, xticks)
pvalues, indicator_results = perform_segment_analysis(t, r,
    indicators, change_metrics, 200.0, dt, 1_000, t_transitions, early_margins)
plot_segment_analysis!(axs, pvalues, t_transitions, indicator_results, early_margins)
fig

#=
From 16 predictions which should all be negative, 10 of them are positive for the variance and 3 are positive for AC1. This clearly points out that generating a reasonable amount of true positives comes here at the expense of generating many false positives. We draw attention upon the fact that **checking for false positives is too seldomly performed in literature!** This does not mean that the method are incorrect but rather that the NGRIP data might be too much of a challenge for a CSD analysis. This is not surprising since the $\delta^{18}O$ time series is noisy and sparsely re-sampled. Furthermore, interpolating over time introduces a potential bias in the statistics, even if performed on a coarse grid. Meh. Is the situation hopeless? Not completely, as we will see in the next section.

!!! info "Future improvement"
    Supporting the computations for uneven time series is a planned improvement of TransitionsInTimeseries. This will avoid the need of regridding data on coarse grids and will prevent from introducing any bias.

## Hindcasting simulated DO-events

In CLIMBER-X[^Willeit2022], an Earth Model of Intermediate Complexity (EMIC), DO-like events can be triggered by forcing the North Atlantic with a (white noise) freshwater input. Simulated DO-like events present the big advantage of being evenly sampled in time and free of measurement noise. Unlike the segment analysis as the one performed above, the analysis fully relying on sliding windows (as introduced in the [Tutorial](@ref)) is a way to generate true positives while simultaneously checking for false positives, since transition metrics are computed for (almost) all time steps. We run this analysis over two exemplary simulation outputs:
=#

opts = [(color = (:orange, 0.1), linewidth = 3),
        (color = Cycled(2), linewidth = 3, linestyle = :dash)]

function perform_sliding_analysis(t, r, indicators, change_metrics, itime, ctime,
    istride, cstride, dt, n)
    config = WindowedIndicatorConfig(indicators, change_metrics; whichtime = last,
        width_ind = Int(itime ÷ dt), stride_ind = istride,
        width_cha = Int(ctime ÷ dt), stride_cha = cstride)
    results = transition_metrics(config, r, t)
    sigconfig = SurrogatesConfig(n = n, tail = :right)
    signif = estimate_significance(sigconfig, results)
    return signif.pvalues, results
end

function plot_sliding_analysis!(axs, pvalues, results, threshold)
    for j in axes(pvalues, 2)
        vlines!(axs[1], results.t_change[pvalues[:, j] .< threshold]; opts[j]...)
        lines!(axs[2+j], results.t_indicator, results.x_indicator[:, j], color = Cycled(1))
        vlines!(axs[2+j], results.t_change[pvalues[:, j] .< threshold]; opts[j]...)
    end
    return nothing
end

t_transitions = [[1850, 2970, 3970, 5070, 5810, 7050, 8050], [3500, 4400, 5790, 7200, 8140]]
figvec = Figure[]
for i in 1:2
    ## Download the data and perform loess filtering on it
    tmp = Base.download("https://raw.githubusercontent.com/JuliaDynamics/JuliaDynamics/" *
        "master/timeseries/climberx-do$(i)-omaxa.csv")
    data = readdlm(tmp)
    tcx, xcx = data[1, :], data[2, :]
    t, x, xtrend, r = loess_filter(tcx, xcx, span = 0.02)

    ## Initialize figure
    xlims = (0, last(tcx))
    xticks = kyr_xticks(xlims[1]:1e3:xlims[2])
    fig, axs = plot_do(tcx, xcx,  t, xtrend, t, r, t_transitions[i], xlims, xticks)
    ylims!(axs[1], (5, 40))
    axs[1].ylabel = L"Max. Atlantic overturning (Sv) $\,$"

    ## Run sliding analysis and update figure with results
    dt = mean(diff(tcx))
    pvalues, results = perform_sliding_analysis(t, r, indicators, change_metrics,
        500.0, 20.0, 1, 1, dt, 1_000)
    plot_sliding_analysis!(axs, pvalues, results, 0.01)
    vlines!(axs[1], t_transitions[i], color = Cycled(1), linewidth = 3)
    push!(figvec, fig)
end
figvec[1]

#=
We here notice that transitions are always preceeded (early enough but not too early) by a significant increase of AC1, sometimes accompanied by the variance. Diagnostics seem to be very reliable, since all transitions lead to both indicators showing significant increases, synchronously and/or subsequently to the transition. False positives continue to arise, however with much lower ratio than on the NGRIP data. It seems like clean data already allows much better results! To make sure we were not simply lucky, let's look at another simulation:
=#

figvec[2]

#=
Things look equally good here! Assuming that these simulations capture the DO dynamics well, one can hope that higher resolution data with less noise allows to robustly predict and/or diagnose DO events. We here draw attention upon the fact that even $dt = 1 \, \mathrm{yr}$ is a relatively sparse sampling for a physical, simulated process displaying transitions with a quasi-period of $T \in [1000, 1500] \, \mathrm{yr}$.

[^Boers2018]:
    Niklas Boers (2018): [Early-warning signals for Dansgaard-Oeschger events in a high-resolution ice core record](https://doi.org/10.1038/s41467-018-04881-7)

[^Willeit2022]:
    Matteo Willeit et al. (2022): [The Earth system model CLIMBER-X v1.0 – Part 1: Climate model description and validation​​​​​​​​​​​​​​](https://doi.org/10.5194/gmd-15-5905-2022)
=#