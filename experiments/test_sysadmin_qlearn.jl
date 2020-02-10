using POMDPs
using POMDPSimulators
using MCTS

include("iql.jl")
using .IQLearning


include("sysadmin.jl")
sim = RolloutSimulator(max_steps=100, rng=MersenneTwister(7))
mdp = BiSysAdmin()
@show coord_graph_adj_mat(mdp)
exp_policy = JointEpsGreedyPolicy(mdp, 0.01, [MersenneTwister(i+42) for i in 1:n_agents(mdp)])
solver = IQLearningSolver(exp_policy; learning_rate=0.01)
policy = solve(solver, mdp)
# planner = solve(solver, mdp)
# action(planner, initialstate(mdp))

# interesting_states = [
#     initialstate(mdp),
#     MachineState[MachineState(3, 1) for i in 1:4], # all status = dead, load = loaded => [1, 1, 1, 1] should reboot all
#     MachineState[MachineState(3, 1), MachineState(1, 1), MachineState(1, 1), MachineState(1, 1)],
#     MachineState[MachineState(1, 1), MachineState(3, 1), MachineState(1, 1), MachineState(1, 1)],
#     MachineState[MachineState(1, 1), MachineState(1, 1), MachineState(3, 1), MachineState(1, 1)],
#     MachineState[MachineState(1, 1), MachineState(2, 1), MachineState(2, 1), MachineState(2, 1)]
# ]

# #Note: uncomment for running on all states
# res = Dict{statetype(mdp), actiontype(mdp)}()
#     #for s in POMDPs.states(mdp)
# for s in interesting_states
#     a = action(planner, s)
#     res[s] = a
# end
