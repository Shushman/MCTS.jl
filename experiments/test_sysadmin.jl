using POMDPs
using POMDPSimulators
using MCTS

include("sysadmin.jl")
sim = RolloutSimulator(max_steps=100, rng=MersenneTwister(7))
mdp = BiSysAdmin()
@show coord_graph_adj_mat(mdp)
d=20; n=200; c=10.
@show d, n, c
solver = MCTSSolver(depth=d, n_iterations=n, exploration_constant=c, rng=MersenneTwister(8))

planner = solve(solver, mdp)
action(planner, initialstate(mdp))

interesting_states = [
    initialstate(mdp),
    MachineState[MachineState(3, 1) for i in 1:4], # all status = dead, load = loaded => [1, 1, 1, 1] should reboot all
    MachineState[MachineState(3, 1), MachineState(1, 1), MachineState(1, 1), MachineState(1, 1)],
    MachineState[MachineState(1, 1), MachineState(3, 1), MachineState(1, 1), MachineState(1, 1)],
    MachineState[MachineState(1, 1), MachineState(1, 1), MachineState(3, 1), MachineState(1, 1)],
    MachineState[MachineState(1, 1), MachineState(2, 1), MachineState(2, 1), MachineState(2, 1)]
]

#Note: uncomment for running on all states
res = Dict{statetype(mdp), actiontype(mdp)}()
    #for s in POMDPs.states(mdp)
for s in interesting_states
    a = action(planner, s)
    res[s] = a
end
