# # Examples for TransitionsInTimeseries.jl

# ## Permutation entropy

# Permutation entropy is used frequently to detect a transition between
# one dynamic regime to another. It is useful when the mean and std. of the
# timeseries values are very similar between the two regimes, which would mean that
# common distribution-based indicators, or common critical-slowing-down based indicators,
# would fail. A simple example of this is a transition from periodic to chaotic motion in
# the logistic map. First, let's generate a timeseries of the logistic map
# that continuously transitions from periodic to chaotic motion

using DynamicalSystemsBase
using CairoMakie

logistic_rule(u, p, t) = @inbounds SVector(p[1]*u[1]*(1 - u[1]))
ds = DeterministicIteratedMap(logistic_rule, [0.5], [1.0])

r1 = 3.6
r2 = 3.9
N = 1000
rs = range(r1, r2; length = N)
x = zeros(N)
for (i, r) in enumerate(rs)
    set_parameter!(ds, 1, r)
    step!(ds)
    x[i] = current_state(ds)[1]
end


# N = 50
# set_parameter!(ds, 1, 3.83)
# tr, t = trajectory(ds, N)
# set_parameter!(ds, 1, 3.85)
# tr2, t = trajectory(ds, N)
# append!(tr, tr2)
# x = tr[:, 1]
lines(x)
# rs = 3.60:0.0001:3.9

# for r in rs
#     set_parameter!(ds, 1, r)
#     for i in 1:3
#         DynamicalSystems.step!(integ)
#         push!(x, integ.u)
#     end
# end

# Now, let's compute and plot various statistical indicators and
indicator = PermutationEntropy()
