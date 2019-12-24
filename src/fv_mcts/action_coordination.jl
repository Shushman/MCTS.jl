## Variable Elimination with the Coordination Graph

# This is just for retrieving the best action at the end of planning
function varel_action(mdp::JointMDP{S,A}, tree::JointMCTSTree, s::AbstractVector{S}) where {S,A}

    n_agents = length(s)
    best_action_idxs = MVector{n_agents,Int64}(undef)

    # NOTE: looking up with vector of states
    state_q_stats = tree.q_component_stats[s]

    # Maintain set of potential functions
    # NOTE: Hashing a vector here
    potential_fns = Dict{Vector{Int64},Vector{Float64}}()
    for (comp, q_stats) in zip(tree.coord_graph_components, state_q_stats)
        potential_fns[comp] = q_stats
    end

    # Need this for reverse process
    # Maps agent to other elements in best response functions and corresponding set of actions
    # E.g. Agent 2 -> (3,4) in its best response and corresponding vector of agent 2 best actions
    best_response_fns = Dict{Int64,Tuple{Vector{Int64},Vector{Int64}}}()

    # Iterate over variable ordering
    # Need to maintain intermediate tables
    for ag_idx in tree.min_degree_ordering

        # Lookup factors with agent in them and simultaneously construct
        # members of new potential function, and delete old factors
        agent_factors = Vector{Vector{Int64}}(undef, 0)
        new_potential_members = Vector{Int64}(undef, 0)
        for k in collect(keys(potential_fns))
            if ag_idx in k
                
                # Agent to-be-eliminated is in factor
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

        if isempty(new_potential_members) == true 

            # No out neighbors..either at beginning or end of ordering
            @assert agent_factors == [[ag_idx]] "agent_factors $(agent_factors) is not just [ag_idx] $([ag_idx])!"

            _, best_ac_idx = findmax(potential_fns[[ag_idx]])
            best_action_idxs[ag_idx] = best_ac_idx

        else

            # Generate new potential function
            # AND the best response vector for eliminated agent
            n_comp_actions = prod([length(tree.agent_actions[c]) for c in new_potential_members])
            comp_tup = Tuple(1:length(tree.agent_actions[c]) for c in new_potential_members)

            # Initialize q-stats for new potential and best response action vector
            # will be inserted into corresponding dictionaries at the end
            new_potential_stats = Vector{Float64}(undef, n_comp_actions)
            best_response_vect = Vector{Int64}(undef, n_comp_actions)

            # Iterate over new potential joint actions and compute new payoff and best response
            for comp_ac_idx = 1:n_comp_actions

                # Get joint action for other members in potential
                ct_idx = CartesianIndices(comp_tup)[comp_ac_idx]

                # For maximizing over agent actions
                ag_ac_values = zeros(MVector{length(tree.agent_actions[ag_idx])})

                # TODO: Agent actions should already be in order
                for ag_ac_idx = 1:length(tree.agent_actions[ag_idx])

                    # Need to look up corresponding stats from agent_factors
                    for factor in agent_factors

                        # NOTE: Need to reconcile the ORDER of ag_idx in factor
                        factor_action_idxs = MVector{length(factor),Int64}(undef)

                        for (idx, f) in enumerate(factor)

                            # if f is ag_idx, set corresponding factor action to ag_ac
                            if f == ag_idx
                                factor_action_idxs[idx] = ag_ac_idx
                            else
                                # Lookup index for corresp. agent action in ct_idx
                                new_pot_idx = findfirst(isequal(f), new_potential_members)
                                factor_action_idxs[idx] = ct_idx[new_pot_idx]
                            end # f == ag_idx
                        end

                        # NOW we can look up the stats of the factor
                        factor_tup = Tuple(1:length(tree.agent_actions[c]) for c in factor)
                        factor_action_linidx = LinearIndices(factor_tup)[factor_action_idxs...]

                        # TODO: verify for correctness. This is looking up the q-stats value of the factor action
                        # and adding to ag_ac_values at the corresp idx.
                        ag_ac_values[ag_ac_idx] += potential_fns[factor][factor_action_linidx]
                    end # factor in agent_factors
                end # (ag_ac_idx, ag_ac) in tree.agent_actions


                # Now we lookup ag_ac_values for the best value to be put in new_potential_stats
                # and the best index to be put in best_response_vect
                best_val, best_idx = findmax(ag_ac_values)

                new_potential_stats[comp_ac_idx] = best_val
                best_response_vect[comp_ac_idx] = best_idx
            end # comp_ac_idx in n_comp_actions

            # Finally, we enter new stats vector and best response vector back to dicts
            potential_fns[new_potential_members] = new_potential_stats
            best_response_fns[ag_idx] = (new_potential_members, best_response_vect)
        end # isempty(new_potential_members)

        # Delete keys in agent_factors from potential fns since variable has been eliminated
        for factor in agent_factors
            delete!(potential_fns, factor)
        end
    end # ag_idx in min_deg_ordering

    # NOTE: At this point, best_action_idxs has at least one entry...for the last action obtained
    @assert sum(isassigned(best_action_idxs, i) for i = 1:n_agents) >= 1 "best_action_idxs is still undefined!"

    # Message passing in reverse order to recover best action
    for ag_idx in Base.Iterators.reverse(tree.min_degree_ordering)

        # Only do something if best action already not obtained
        if isassigned(best_action_idxs, ag_idx) == false

            # Should just be able to lookup best response function
            (agents, best_response_vect) = best_response_fns[ag_idx]

            # Members of agents should already have their best action defined
            agent_ac_tup = Tuple(1:length(tree.agent_actions[c]) for c in agents)
            best_agents_action_idxs = [best_action_idxs[ag] for ag in agents]
            best_response_idx = LinearIndices(agent_ac_tup)[best_agents_action_idxs...]

            # Assign best action for ag_idx
            best_action_idxs[ag_idx] = best_response_idx
        end # isdefined
    end

    # Finally, return best action by iterating over best action indices
    best_action = [tree.agent_actions[ag][idx] for (ag, idx) in enumerate(best_action_idxs)]

    return best_action
end

# This is to compute the best action for UCB
function varel_action_UCB end

function varel_value end
