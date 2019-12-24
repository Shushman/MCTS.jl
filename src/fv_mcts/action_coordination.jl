## Variable Elimination with the Coordination Graph

# This is just for retrieving the best action at the end of planning
function varel_action(mdp::JointMDP{S,A}, tree::JointMCTSTree, s::AbstractVector{S}) where {S,A}

    n_agents = length(s)
    best_action = Vector{A}(undef, n_agents) # TODO: More generic

    # NOTE: looking up with vector of states
    state_q_stats = tree.q_component_stats[s]

    # Maintain set of potential functions
    # TODO: Can't do this in practice
    potential_fns = Dict{Vector{Int64},Vector{Float64}}()
    for (comp, q_stats) in zip(tree.coord_graph_components, state_q_stats)
        potential_fns[comp] = q_stats
    end

    # Need this for reverse process
    best_response_fns = Dict{Vector{Int64},Vector{A}}()

    # Iterate over variable ordering
    # Need to maintain intermediate tables
    for ag_idx in tree.min_degree_ordering

        # Lookup factors with agent in them
        agent_factors = Vector{Vector{Int64}}(undef, 0)
        new_potential_members = Vector{Int64}(undef, 0)
        for k in collect(keys(potential_fns))
            if ag_idx in k
                push!(agent_factors, k)

                # Construct key for new potential as union of all others
                # except ag_idx
                for ag in k
                    if ag != ag_idx
                        push!(new_potential_members, ag)
                    end
                end
            end
        end

        if isempty(new_potential_members) == true # No out neighbors..either at beginning or end

            @assert agent_factors == [[ag_idx]]

            _, best_ac_idx = findmax(potential_fns[[ag_idx]])

            # TODO: Review for correctness
            best_action[ag_idx] = tree.agent_actions[ag_idx][best_ac_idx]
        else

            # Hard part - generate new potential function
            # AND the best response vector for eliminated agent

        end

    end # ag_idx in min_deg_ordering

    # Message passing in reverse order to recover best action
    for ag_idx in Base.Iterators.reverse(tree.min_degree_ordering)

        # Only do something if best action already not obtained
        if isdefined(best_action, ag_idx) == false

        end # isdefined
    end

    return best_action
end

# This is to compute the best action for UCB
function varel_action_UCB end

function varel_value end
