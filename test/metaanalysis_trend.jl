using TransitionIndicators, Test, Random, TimeseriesSurrogates
using Distributions

@testset "checking slope estimation" begin
    d = Normal()                # define normal distribution
    m, p = 10 .* rand(d, 2)     # slope and offset parameters
    x = 0:0.1:10
    y = m .* x .+ p             # define affine function
    y_noisy = y + 0.1 .* rand(d, length(x))
    m_est, p_est = ridge(x, y)
    m_est_noisy, p_est_noisy = ridge(x, y_noisy)
    @test isapprox(m, m_est)
    @test isapprox(p, p_est)
    @test isapprox(m, m_est_noisy, atol = 1e-1)
    @test isapprox(p, p_est_noisy, atol = 1e-1)
end

@testset "sliding trend estimation over indicator" begin
    n = 1001
    t = collect(1.0:n)
    x = copy(t)
    p = init_metaanalysis_params(n_surrogates = 100)
    res = analyze_indicators(t, x, [mean, var], ridge_slope, p)

    # The trend of mean(windowview) is the stride for x=t
    meantrend_ground_truth = fill(p.wv_indicator_stride, length(res.t_evolution))
    # The trend of var(windowview) is 0 for x any affine function of t.
    vartrend_ground_truth = fill(0.0, length(res.t_evolution))
    @test isapprox(res.X_evolution[1, :, 1], meantrend_ground_truth)
    @test isapprox(res.X_evolution[1, :, 2], vartrend_ground_truth, atol = 1e-12)
end

#= A small  benchmark:
n = 1001
t = collect(1.0:n)
x = copy(t)
m = precompute_ridge(t)[1,:]
@btime ridge_slope($t, $x)
@btime precomputed_ridge_slope($x, $m)

Result on Jan's machine:
  8.033 μs (16 allocations: 41.42 KiB)
  73.909 ns (0 allocations: 0 bytes)
=#
