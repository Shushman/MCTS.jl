## Paradigm 1
# We let the user define the full joint MDP and request for components
"""
    function n_agents(m::MDP)

Return the number of agents in the (PO)MDP
"""
function n_agents end

"""
    function get_coord_graph_components(m::MDP)

Returns a Vector{Set{Int64}} with the subsets of agents in each component of
the coordination graph
"""
function get_coord_graph_components end


"""
    function get_action_component(m::MDP, a::Vector{A}, component::Int64) where A

Returns the index of the subchunk of full joint action a
"""
function get_action_component_index end

"""
Same but for state
"""
function get_state_component_index end

####################################################################################

## Paradigm 2: The user provides individual S,A specs and we maintain the MMDP
# where the S is actually a Vector{S}
# NOTE: Ultimately the gen(s::Vector{S}, a::Vector{A} etc.) function will have to be defined by the user


@with_kw struct JointMDP{N,S,A} <: POMDPs.MDP{SVector{N,S},SVector{N,A}}
    base_mdp::POMDPs.MDP{S,A}   # Can a derived type have a member of the same abstract type?
    coord_graph_components::Vector{Set{Int64}}
end

function create_joint_mdp(num_agents::Int64, base_mdp::POMDPs.MDP{S,A},
                          coord_graph_components::Vector{Set{Int64}}) where {S,A}
    return JointMDP{num_agents,S,A}(base_mdp=base_mdp,
                                    coord_graph_components=coord_graph_components)
end

#
