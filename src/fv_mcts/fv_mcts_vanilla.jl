## We can use the following directly without modification
## 1. domain_knowledge.jl for Rollout, init_Q and init_N functions
## 2. MCTSSolver for representing the overall MCTS (underlying things will change)


# JointMCTS tree has to be different, to efficiently encode Q-stats
# SV is the hashtype for state vector
mutable struct JointMCTSTree{S,SV}

    # To track if state node in tree already
    # NOTE: We don't strictly need this at all if no tree reuse...
    state_map::Dict{SV,Int64}

    # these vectors have one entry for each state node
    # Only doing factored satistics (for actions), not state components
    child_ids::Vector{Vector{Int}}
    total_n::Vector{Int}
    s_labels::Vector{AbstractVector{S}}

    # Track stats for all action components over the n_iterations
    agent_actions::Vector{Int64}
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
    agent_actions = Vector{Int64}(undef, n_agents(joint_mdp))
    for i = 1:n_agents(joint_mdp)
        agent_actions[i] = get_agent_actions(joint_mdp, i)
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

        comp_tup = Tuple(1:length(agent_actions[c]) for c in comp)

        for comp_ac_idx = 1:n_comp_actions

            # Generate action subcomponent and call init_Q and init_N for it
            ct_idx = CartesianIndices(comp_tup)[comp_ac_idx] # Tuple corresp to
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

# Reset tree.
function clear_tree!(planner::JointMCTSPlanner, init_state)

    # Clear out state hash dict entirely
    empty!(planner.tree.state_map)

    # Empty state vectors with state hints
    sz = min(planner.solver.n_iterations, 100_000)

    empty!(planner.tree.child_ids)
    sizehint!(planner.tree.child_ids, planner.solver.n_iterations)

    empty!(planner.tree.total_n)
    sizehint!(planner.tree.total_n, planner.solver.n_iterations)

    empty!(planner.tree.s_labels)
    sizehint!(planner.tree.s_labels, planner.solver.n_iterations)

    # Don't touch coord graph components but reset stats to init_N and init_Q
    # TODO: Code reuse with JointMCTSTree constructor...figure out
    # Also, this assumes that actions really do depend on state. Can be more efficient if not
    fill!(planner.tree.agent_actions, 0)
    for i = 1:n_agents(planner.mdp)
        planner.tree.agent_actions[i] = get_agent_actions(planner.mdp, i)
    end

    for (idx, comp) in enumerate(coord_graph_components)

        n_comp_actions = prod([length(planner.tree.agent_actions[c]) for c in comp])

        planner.tree.n_component_stats[idx] = Vector{Int64}(undef, n_comp_actions)
        planner.tree.q_component_stats[idx] = Vector{Float64}(undef, n_comp_actions)

        comp_tup = Tuple(1:length(planner.tree.agent_actions[c]) for c in comp)

        for comp_ac_idx = 1:n_comp_actions

            # Generate action subcomponent and call init_Q and init_N for it
            ct_idx = CartesianIndices(comp_tup)[comp_ac_idx] # Tuple corresp to
            local_action = [planner.tree.agent_actions[c] for c in Tuple(ct_idx)]

            # NOTE: init_N and init_Q are functions of component AND local action
            for (ag, ac) in zip(comp, local_action)
                n_component_stats[idx][comp_ac_idx] = init_N(joint_mdp, init_state, comp, local_action)
                q_component_stats[idx][comp_ac_idx] = init_Q(joint_mdp, init_state, comp, local_action)
            end # (ag, ac) in zip
        end # comp_ac_idx in n_comp_actions
    end # for enumerate coord_graph_components
end

function get_state_node(tree::JointMCTSTree, s, planner::JointMCTSPlanner)
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
# NOTE: Hardcoding no tree reuse for now
function POMDPs.action(planner::JointMCTSPlanner, s)
    clear_tree!(planner, s) # Always call this at the top
    plan!(planner, s)
    return varel_action(planner.mdp, planner.tree) # TODO: Need to implement
end

## Not implementing value functions right now....
## ..Is it just the MAX of the best action, rather than argmax???

# Could reuse plan! from vanilla.jl. But I don't like
# calling an element of an abstract type like AbstractMCTSPlanner
function plan!(planner::JointMCTSPlanner, s)
    planner.tree = build_tree(planner, s)
end

# Build_Tree can be called on the assumption that no reuse AND tree is reinitialized
function build_tree(planner::JointMCTSPlanner, s)

    n_iterations = planner.solver.n_iterations
    depth = planner.solver.depth

    root = insert_node!(tree, planner, s)

    # build the tree
    for n = 1:n_iterations
        simulate(planner, root, depth)
    end
    return tree
end

function simulate(planner::JointMCTSPlanner, node::StateNode, depth::Int64)

    mdp = planner.mdp
    rng = planner.rng
    s = state(node)
    tree = node.tree

    # once depth is zero return
    if isterminal(planner.mdp, s)
       return 0.0
    elseif depth == 0
        return estimate_value(planner.solved_estimate, planner.mdp, s, depth)
    end

    # Choose best UCB action (NOT an action node)
    ucb_action = varel_action_UCB(mdp, planner.tree, planner.solver.exploration_constant)

    # MC Transition
    sp, r = gen(DDNOut(:sp, :r), mdp, s, ucb_action, rng)

    spid = get(tree.state_map, hash(sp), 0) # may be non-zero even with no tree reuse
    if spid == 0
        spn = insert_node!(tree, planner, sp)
        spid = spn.id
        q = r + discount(mdp) * estimate_value(planner.solved_estimate, planner.mdp, sp, depth - 1)
    else
        q = r + discount(mdp) * simulate(planner, StateNode(tree, spid) , depth - 1)
    end

    ## Not bothering with tree vis right now

    tree.total_n[node.id] += 1

    # Update component statistics! (non-trivial)
    # This is related but distinct from initialization
    for (idx, comp) in enumerate(tree.coord_graph_components)

        # Create cartesian index tuple
        comp_tup = Tuple(1:length(tree.agent_actions[c]) for c in comp)

        # RECOVER local action corresp. to ucb action
        # TODO: Review this carefully. Need @req for action index for agent.
        local_action = [ucb_action[c] for c in comp]
        local_action_idxs = [actionindex(mdp, a, c) for (a, c) in zip(local_action, comp)]

        # Is unrolling the right thing here?
        comp_ac_idx = LinearIndices(comp_tup)[local_action_idxs...]

        # NOTE: NOW we can update stats. Could generalize incremental update more here
        tree.n_component_stats[idx][comp_ac_idx] += 1
        tree.q_component_stats[idx][comp_ac_idx] += (q - tree.q_component_stats[idx][comp_ac_idx]) / tree.n_component_stats[idx][comp_ac_idx]
    end
    return q
end
