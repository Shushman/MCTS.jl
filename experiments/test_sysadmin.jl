using POMDPs
using POMDPSimulators
using MCTS

sim = RolloutSimulator(max_steps=100, rng=MersenneTwister(7))
mdp = UniSysAdmin()
d=20; n=100; c=10.
@show d, n, c
solver = MCTSSolver(depth=d, n_iterations=n, exploration_constant=c, rng=MersenneTwister(8))

planner = solve(solver, mdp)
simulate(sim, mdp, planner)
