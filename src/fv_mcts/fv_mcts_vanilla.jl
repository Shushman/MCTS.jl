## We can use the following directly without modification
## 1. domain_knowledge.jl for Rollout, init_Q and init_N functions
## 2. MCTSSolver for representing the overall MCTS (underlying things will change)


# JointMCTS tree has to be different, to efficiently encode Q-stats
# SV is the hashtype for state vector
mutable struct JointMCTSTree{S,SV}

    # To track if state node in tree already
    # TODO: Is this acceptable? I prefer this so I can explicitly track the vector type
    state_map::Dict{SV,Int64}

    # these vectors have one entry for each state node
    # Only doing factored satistics (for actions), not state components
    child_ids::Vector{Vector{Int}}
    total_n::Vector{Int}
    s_labels::Vector{AbstractVector{S}}

    # Track stats for all action components over the n_iterations
    coord_graph_components::Vector{Vector{Int64}}
    n_component_stats::Vector{Vector{Int64}}
    q_component_stats::Vector{Vector{Float64}}

    # Don't need a_labels because need to do var-el for best action anyway
end

# coord_graph_comps given by JointMCTSPlanner
# NOTE: This is explicitly assuming no tree reuse
function JointMCTSTree(joint_mdp::JointMDP{S,A},
                       coord_graph_components::Vector{Vector{Int64}},
                       init_state::AbstractVector{S},
                       sz::Int64=1000) where {S, A}

    # TODO: add to requirements
    svht = state_vec_hash_type(joint_mdp)

    # Initialize full agent actions
    # NOTE: This can be more efficient if all agents have same actions...still, it's linear not exp
    agent_actions = Vector{Int64}(undef, n_agents(joint_mdp))
    for i = 1:n_agents(joint_mdp)
        agent_actions[i] = get_agent_actions(joint_mdp, i, init_state)
    end

    # Initialize component stats
    n_comps = length(coord_graph_components)
    n_component_stats = Vector{Vector{Int64}}(undef, n_comps)
    q_component_stats = Vector{Vector{Float64}}(undef, n_comps)

    # Iterate over components and initialize
    for (idx, comp) in enumerate(coord_graph_components)

        n_comp_actions = prod([length(agent_actions[c]) for c in comp])

        n_component_stats[idx] = Vector{Int64}(undef, n_comp_actions)
        q_component_stats[idx] = Vector{Float64}(undef, n_comp_actions)

        comp_tup = Tuple(1:n_agent_actions[c] for c in comp)

        for comp_ac_idx = 1:n_comp_actions

            # Generate action subcomponent and call init_Q and init_N for it
            ct_idx = CartesianIndices(tup)[comp_ac_idx] # Tuple corresp to
            local_action = [agent_actions[c] for c in Tuple(ct_idx)]

            # NOTE: init_N and init_Q are functions of component AND local action
            for (ag, ac) in zip(comp, local_action)
                n_component_stats[idx][comp_ac_idx] = init_N(joint_mdp, init_state, comp, local_action)
                q_component_stats[idx][comp_ac_idx] = init_Q(joint_mdp, init_state, comp, local_action)
            end

        end # comp_ac_idx in n_comp_actions
    end # for comp in n_comps


    return JointMCTSTree{S,svht}(Dict{svht,Int64}(),

                                sizehint!(Vector{Int}[], sz),
                                sizehint!(Int[], sz),
                                sizehint!(S[], sz),

                                coord_graph_components,
                                n_component_stats,
                                q_component_stats
                                )
end # function



Base.isempty(t::JointMCTSTree) = isempty(t.state_map)
state_nodes(t::JointMCTSTree) = (JointStateNode(t, id) for id in 1:length(t.total_n))

struct JointStateNode{S,SV}
    tree::JointMCTSTree{S,SV}
    id::Int64
end

get_state_node(tree::JointMCTSTree, s) = StateNode(tree, s)

# accessors for state nodes
@inline state(n::JointStateNode) = n.tree.s_labels[n.id]
@inline total_n(n::JointStateNode) = n.tree.total_n[n.id]
@inline child_ids(n::JointStateNode) = n.tree.child_ids[n.id]

## No need for `children` or ActionNode just yet

mutable struct JointMCTSPlanner{S, A, SV, SE, RNG <: AbstractRNG} <: AbstractMCTSPlanner{JM}
    solver::MCTSSolver
    mdp::JointMDP{S,A}
    tree::JointMCTSTree{S,SV}
    solved_estimate::SE
    rng::RNG
end

# NOTE: JointMCTS Planner gets the adjacency matrix as argument
function JointMCTSPlanner(solver::MCTSSolver,
                          mdp::JointMDP{S,A},
                          coord_adj_mat::Matrix{Int64},
                          init_state::S
                         ) where {S,A}

    # Get coord graph comps from maximal cliques of graph
    @assert size(coord_adj_mat)[1] == n_agents(mdp) "Adjacency Mat does not match number of agents!"
    coord_graph_components = maximal_cliques(SimpleGraph(coord_adj_mat))

    # Create tree FROM CURRENT STATE
    tree = JointMCTSTree(mdp, coord_graph_components, init_state, solver.n_iterations)
    se = convert_estimator(solver.estimate_value, solver, mdp)

    return JointMCTSPlanner(solver, mdp, tree, se, solver.rng)
end # end JointMCTSPlanner

function clear_tree!(p::JointMCTSPlanner) p.tree = nothing end

function get_state_node(tree::JointMCTSTree, s, planner::JointMCTSPlanner)
    # TODO: Add hash to req
    shash = hash(s)

    if haskey(tree.state_map, shash)
        return StateNode(tree, tree.state_map[shash]) # Is this correct? Not equiv to vanilla
    else
        return insert_node!(tree, planner, s)
    end
end

# no computation is done in solve - the solver is just given the mdp model that it will work with
POMDPs.solve(solver::MCTSSolver, mdp::JointMDP) = JointMCTSPlanner(solver, mdp)

# IMP: Overriding action for JointMCTSPlanner here
function POMDPs.action(p::JointMCTSPlanner, s)
    tree = plan!(p, s)
    return varel_action(tree) # TODO: Need to implement
end

## Not implementing value functions right now....
## ..Is it just the MAX of the best action, rather than argmax???
