## Paradigm 1
# We let the user define the full joint MDP and request for
abstract type JointMDP{S,A} <: POMDPs.MDP{AbstractVector{S},AbstractVector{A}} end

"""
    function n_agents(m::JointMDP)

Return the number of agents in the (PO)MDP
"""
function n_agents end

"""
    function get_agent_actions (m::JointMDP, idx::Int64, s::AbstractVector{S}) where S

Returns the discrete actions for the given agent index
"""
# NOTE: This will be called a LOT so it should not allocate each time....
function get_agent_actions end


"""
    function coord_graph_adj_mat(m::JointMDP)

Returns the Matrix{Int64} for the coordination graph. (this will be sparsified and converted to a coord graph)
N.B. If it is a fully connected graph, just return ones(Int64, n_agents, n_agents)?
"""
function coord_graph_adj_mat end

# NOTE: JKG, Continue from here
# User should override statetype and actiontype to return the CONCRETE type of state
init_Q(n::Number, mdp::JointMDP, s, c, a) = convert(Float64, n)
init_N(n::Number, mdp::JointMDP, s, c, a) = convert(Int, n)


function POMDPs.actions(p::JointMDP{S, A}, s) where {S, A}
    collect(Iterators.product((get_agent_actions(p, i, s) for i in 1:n_agents(p))...))
end

function POMDPs.actions(p::JointMDP{S, A}) where {S, A}
    collect(Iterators.product((get_agent_actions(p, i) for i in 1:n_agents(p))...))
end

function POMDPs.simulate(sim::RolloutSimulator, mdp::JointMDP, policy::Policy, initialstate::S) where {S}

    if sim.eps == nothing
        eps = 0.0
    else
        eps = sim.eps
    end

    if sim.max_steps == nothing
        max_steps = typemax(Int)
    else
        max_steps = sim.max_steps
    end

    s = initialstate

    disc = 1.0
    r_total = zeros(n_agents(mdp))
    step = 1

    while disc > eps && !isterminal(mdp, s) && step <= max_steps
        a = action(policy, s)

        sp, r = gen(DDNOut(:sp,:r), mdp, s, a, sim.rng)

        r_total .+= disc.*r

        s = sp

        disc *= discount(mdp)
        step += 1
    end

    return r_total
end
