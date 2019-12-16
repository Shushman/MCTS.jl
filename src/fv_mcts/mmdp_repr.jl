## Paradigm 1
# We let the user define the full joint MDP and request for
abstract type JointMDP{S,A} <: POMDPs.MDP{AbstractVector{S},AbstractVector{A}} end

"""
    function n_agents(m::JointMDP)

Return the number of agents in the (PO)MDP
"""
function n_agents end

"""
    function agent_actions (m::JointMDP, idx::Int64, s::AbstractVector{S}) where S

Returns the discrete actions for the given agent index and the current state
"""
function agent_actions end

"""
    function coord_graph_adj_mat(m::JointMDP)

Returns the Matrix{Int64} for the coordination graph. (this will be sparsified and converted to a coord graph)
"""
function coord_graph_adj_mat end

# NOTE: JKG, Continue from here
