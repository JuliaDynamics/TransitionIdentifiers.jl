module TransitionIndicators

# Use the README as the module docs
@doc let
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end TransitionIndicators

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

include("library/precomputation.jl")
include("library/indicators.jl")
include("library/change_metrics_trend.jl")

include("analysis/analysis_types.jl")
include("analysis/perform_analysis.jl")

# windowing.jl
export WindowViewer, windowmap, windowmap!, midpoint, midvalue

# library
export PrecomputableFunction, precompute_metrics, precompute
export ar1_whitenoise
export LowfreqPowerSpectrum, PrecomputedLowfreqPowerSpectrum
export mean, std, var, skewness, kurtosis       # from StatsBase
export PermutationEntropy
export SymbolicPermutation, entropy_normalized  # from ComplexityMeasures
export kendalltau, spearman
export RidgeRegressionSlope, PrecomputedRidgeRegressionSlope

# analysis
export IndicatorsConfig, SignificanceConfig, IndicatorsResults
export indicators_analysis, indicators_significance

# timeseries
export isequispaced, equispaced_step

# load_data.jl
export load_linear_vs_doublewell

end # module TransitionIndicators