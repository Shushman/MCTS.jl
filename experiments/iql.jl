module IQLearning

using Random
using MCTS: JointMDP, n_agents, get_agent_actions, get_agent_states, get_agent_actionindex, get_agent_stateindex


using POMDPs
import POMDPs: action, actionvalues, solve, updater

using POMDPSimulators

using Infiltrator


struct JointValuePolicy{P<:JointMDP, T, A} <: Policy
    mdp::P
    value_tables::T
    act::A
end

function JointValuePolicy(mdp::JointMDP, value_tables=tuple((zeros(length(get_agent_states(mdp, ag)), length(get_agent_actions(mdp, ag))) for ag in 1:n_agents(mdp))...))
    return JointValuePolicy(mdp, value_tables, tuple((get_agent_actions(mdp, ag) for ag in 1:n_agents(mdp))...))
end


action(p::JointValuePolicy, s) = [p.act[ag][argmax(p.value_tables[ag][get_agent_stateindex(p.mdp, ag, s[ag]), :])] for ag in 1:n_agents(p.mdp)]
actionvalues(p::JointValuePolicy, s) = [p.value_tables[ag][get_agent_stateindex(p.mdp, ag, s[ag]), :] for ag in 1:n_agents(p.mdp)]


struct JointStochasticPolicy{D, RNG <: AbstractRNG} <: Policy
    distributions::D
    rngs::Vector{RNG}
end

function action(policy::JointStochasticPolicy, s)
    return [rand(policy.rngs[ag], policy.distributions[ag]) for ag in 1:length(policy.rngs)]
end

updater(policy::JointStochasticPolicy) = VoidUpdater()

JointUniformRandomPolicy(problem, rngs) = JointStochasticPolicy(tuple((get_agent_actions(problem, ag) for ag in 1:n_agents(problem))...), rngs)

mutable struct JointEpsGreedyPolicy{V,U} <: Policy
    eps::Float64
    val::V
    uni::U
end

JointEpsGreedyPolicy(mdp::JointMDP, eps::Float64, rngs) = JointEpsGreedyPolicy(eps, JointValuePolicy(mdp), JointUniformRandomPolicy(mdp, rngs))

function action(policy::JointEpsGreedyPolicy, s)
    greedy_acts = action(policy.val, s)
    rand_acts = action(policy.uni, s)
    select = rand.(policy.uni.rngs) .> policy.eps
    acts = select .* greedy_acts .+ (1. .- select) .* rand_acts
    return acts
end

mutable struct IQLearningSolver <: Solver
    n_episodes::Int64
    max_episode_length::Int64
    learning_rate::Float64
    exploration_policy::JointEpsGreedyPolicy
    Q_vals::Union{Nothing, Matrix{Float64}}
    eval_every::Int64
    n_eval_traj::Int64
    rng::AbstractRNG
    verbose::Bool
    function IQLearningSolver(exp_policy::JointEpsGreedyPolicy;
                              rng=Random.GLOBAL_RNG,
                              n_episodes=100,
                              Q_vals = nothing,
                              max_episode_length=100,
                              learning_rate=0.001,
                              eval_every=10,
                              n_eval_traj=20,
                              verbose=true)
        return new(n_episodes, max_episode_length, learning_rate, exp_policy, Q_vals, eval_every, n_eval_traj, rng, verbose)
    end
end

function solve(solver::IQLearningSolver, mdp::JointMDP)
    rng = solver.rng

    if solver.Q_vals === nothing
        Qs = tuple((zeros(length(get_agent_states(mdp, i)), length(get_agent_actions(mdp, i))) for i in 1:n_agents(mdp))...)
    else
        Qs = solver.Q_vals
    end
    exploration_policy = solver.exploration_policy
    sim = RolloutSimulator(rng=rng, max_steps=solver.max_episode_length)


    policy = JointValuePolicy(mdp, Qs, tuple((get_agent_actions(mdp, ag) for ag in 1:n_agents(mdp))...))
    exploration_policy.val = policy


    for i = 1:solver.n_episodes
        s = initialstate(mdp, rng)
        t = 0
        while !isterminal(mdp, s) && t < solver.max_episode_length
            a = action(exploration_policy, s)
            @show a
            sp, r = gen(DDNOut(:sp, :r), mdp, s, a, rng)
            for ag in 1:n_agents(mdp)
                si = get_agent_stateindex(mdp, ag, s[ag])
                ai = get_agent_actionindex(mdp, ag, a[ag])
                spi = get_agent_stateindex(mdp, ag, sp[ag])
                Qs[ag][si, ai] += solver.learning_rate * (r[ag] + discount(mdp) * maximum(Qs[ag][spi, :]) - Qs[ag][si,ai])
            end
            s = sp
            t += 1
        end
        if i % solver.eval_every == 0
            r_tot = zeros(n_agents(mdp))
            for traj in 1:solver.n_eval_traj
                r_tot .+= simulate(sim, mdp, policy, initialstate(mdp, rng))
            end
            solver.verbose ? println("On Iteration $i, Returns: $(sum(r_tot)/solver.n_eval_traj)") : nothing
        end
    end
    return policy
end

export IQLearningSolver
export JointEpsGreedyPolicy

end
