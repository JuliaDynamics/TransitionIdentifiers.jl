module TransitionsInTimeseries

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end TransitionsInTimeseries

using LinearAlgebra
using Random
using Downloads
using DelimitedFiles
using InteractiveUtils
using FFTW
using ComplexityMeasures

using Reexport
@reexport using TimeseriesSurrogates

include("misc/params.jl")
include("misc/windowing.jl")
include("misc/timeseries.jl")
include("misc/load_data.jl")
include("misc/precomputation.jl")

include("indicators/critical_slowing_down.jl")
include("indicators/distribution_distance.jl")
include("indicators/nlts.jl")
include("indicators/spectral.jl")
include("indicators/statistics.jl")

include("change_metrics/slope.jl")
include("change_metrics/valuediff.jl")

include("analysis/analysis_types.jl")
include("analysis/perform_analysis.jl")

# windowing.jl
export WindowViewer, windowmap, windowmap!, midpoint, midvalue

# library
export PrecomputableFunction, precompute
export ar1_whitenoise
export LowfreqPowerSpectrum, PrecomputedLowfreqPowerSpectrum
export mean, std, var, skewness, kurtosis # from StatsBase
export entropy, PermutationEntropy
export kendalltau, spearman
export RidgeRegressionSlope, PrecomputedRidgeRegressionSlope
export difference_of_means

# analysis
export TransitionsSurrogatesConfig, estimate_transitions
export transition_flags

# timeseries
export isequispaced, equispaced_step

# load_data.jl
export load_linear_vs_doublewell

end # module TransitionsInTimeseries