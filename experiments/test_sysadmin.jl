using POMDPs
using POMDPSimulators
using MCTS

include("sysadmin.jl")
sim = RolloutSimulator(max_steps=100, rng=MersenneTwister(7))
mdp = BiSysAdmin()
@show coord_graph_adj_mat(mdp)
d=20; n=100; c=10.
@show d, n, c
solver = MCTSSolver(depth=d, n_iterations=n, exploration_constant=c, rng=MersenneTwister(8))

planner = solve(solver, mdp)
action(planner, initialstate(mdp))
