## We can use the following directly without modification
## 1. domain_knowledge.jl for Rollout, init_Q and init_N functions
## 2. MCTSSolver for representing the overall MCTS (underlying things will change)


# JointMCTS tree has to be different, to efficiently encode Q-stats
mutable struct JointMCTSTree{S,A}

    # To track if state node in tree already
    # NOTE: We don't strictly need this at all if no tree reuse...
    state_map::Dict{<:AbstractVector{S},Int64}

    # these vectors have one entry for each state node
    # Only doing factored satistics (for actions), not state components
    # Looks like we don't need child_ids and total_n
    s_labels::Vector{<:AbstractVector{S}}

    # Track stats for all action components over the n_iterations
    agent_actions::Vector{<:AbstractVector{A}}
    coord_graph_components::Vector{Vector{Int64}}
    min_degree_ordering::Vector{Int64}

    n_component_stats::Dict{<:AbstractVector{S},Vector{Vector{Int64}}}
    q_component_stats::Dict{<:AbstractVector{S},Vector{Vector{Float64}}}

    # Don't need a_labels because need to do var-el for best action anyway
end

# coord_graph_comps given by JointMCTSPlanner
# NOTE: This is explicitly assuming no tree reuse
function JointMCTSTree(joint_mdp::JointMDP{S,A},
                       coord_graph_components::Vector{Vector{Int64}},
                       min_degree_ordering::Vector{Int64},
                       init_state::AbstractVector{S},
                       sz::Int64=1000) where {S, A}

    # Initialize full agent actions
    agent_actions = Vector{actiontype(joint_mdp)}(undef, n_agents(joint_mdp))
    for i = 1:n_agents(joint_mdp)
        agent_actions[i] = get_agent_actions(joint_mdp, i)
    end

    return JointMCTSTree{S}(Dict{typeof(init_state),Int64}(),

                                sizehint!(Vector{typeof(init_state)}, sz),

                                agent_actions,
                                coord_graph_components,
                                min_degree_ordering,
                                Dict{typeof(init_state),Vector{Vector{Int64}}}(),
                                Dict{typeof(init_state),Vector{Vector{Int64}}}()
                                )
end # function



Base.isempty(t::JointMCTSTree) = isempty(t.state_map)

struct JointStateNode{S}
    tree::JointMCTSTree{S}
    id::Int64
end

get_state_node(tree::JointMCTSTree, id) = JointStateNode(tree, id)

# accessors for state nodes
@inline state(n::JointStateNode) = n.tree.s_labels[n.id]

## No need for `children` or ActionNode just yet

mutable struct JointMCTSPlanner{S, A, SE, RNG <: AbstractRNG} <: AbstractMCTSPlanner{JointMDP{S,A}}
    solver::MCTSSolver
    mdp::JointMDP{S,A}
    tree::JointMCTSTree{S}
    solved_estimate::SE
    rng::RNG
end

function JointMCTSPlanner(solver::MCTSSolver,
                          mdp::JointMDP{S,A},
                          init_state::AbstractVector{S},
                         ) where {S,A}

    # Get coord graph comps from maximal cliques of graph
    adjmat = coord_graph_adj_mat(mdp)
    @assert size(adjmat)[1] == n_agents(mdp) "Adjacency Mat does not match number of agents!"

    adjmatgraph = SimpleGraph(adjmat)
    coord_graph_components = maximal_cliques(adjmatgraph)
    min_degree_ordering = sortperm(degree(adj))

    # Create tree FROM CURRENT STATE
    tree = JointMCTSTree(mdp, coord_graph_components, min_degree_ordering,
                         init_state, solver.n_iterations)
    se = convert_estimator(solver.estimate_value, solver, mdp)

    return JointMCTSPlanner(solver, mdp, tree, se, solver.rng)
end # end JointMCTSPlanner

# Reset tree.
function clear_tree!(planner::JointMCTSPlanner)

    # Clear out state hash dict entirely
    empty!(planner.tree.state_map)

    # Empty state vectors with state hints
    sz = min(planner.solver.n_iterations, 100_000)

    empty!(planner.tree.s_labels)
    sizehint!(planner.tree.s_labels, planner.solver.n_iterations)

    # Don't touch agent_actions and coord graph component
    # Just clear comp stats dict
    empty!(planner.tree.n_component_stats)
    empty!(planner.tree.q_component_stats)
end

function get_state_node(tree::JointMCTSTree, s, planner::JointMCTSPlanner)
    if haskey(tree.state_map, s)
        return JointStateNode(tree, tree.state_map[s]) # Is this correct? Not equiv to vanilla
    else
        return insert_node!(tree, planner, s)
    end
end

# no computation is done in solve - the solver is just given the mdp model that it will work with
POMDPs.solve(solver::MCTSSolver, mdp::JointMDP) = JointMCTSPlanner(solver, mdp, initialstate(mdp))

# IMP: Overriding action for JointMCTSPlanner here
# NOTE: Hardcoding no tree reuse for now
function POMDPs.action(planner::JointMCTSPlanner, s)
    clear_tree!(planner) # Always call this at the top
    plan!(planner, s)
    return varel_action(planner.mdp, planner.tree, s) # TODO: Need to implement
end

## Not implementing value functions right now....
## ..Is it just the MAX of the best action, rather than argmax???

# Could reuse plan! from vanilla.jl. But I don't like
# calling an element of an abstract type like AbstractMCTSPlanner
function plan!(planner::JointMCTSPlanner, s)
    planner.tree = build_tree(planner, s)
end

# Build_Tree can be called on the assumption that no reuse AND tree is reinitialized
function build_tree(planner::JointMCTSPlanner, s::AbstractVector{S}) where S

    n_iterations = planner.solver.n_iterations
    depth = planner.solver.depth

    root = insert_node!(tree, planner, s)

    # build the tree
    for n = 1:n_iterations
        simulate(planner, root, depth)
    end
    return tree
end

function simulate(planner::JointMCTSPlanner, node::JointStateNode, depth::Int64)

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
    ucb_action = varel_action_UCB(mdp, planner.tree, planner.solver.exploration_constant, s)

    # MC Transition
    sp, r = gen(DDNOut(:sp, :r), mdp, s, ucb_action, rng)

    spid = get(tree.state_map, sp, 0) # may be non-zero even with no tree reuse
    if spid == 0
        spn = insert_node!(tree, planner, sp)
        spid = spn.id
        q = r + discount(mdp) * estimate_value(planner.solved_estimate, planner.mdp, sp, depth - 1)
    else
        q = r + discount(mdp) * simulate(planner, JointStateNode(tree, spid) , depth - 1)
    end

    ## Not bothering with tree vis right now

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
        tree.n_component_stats[s][idx][comp_ac_idx] += 1
        tree.q_component_stats[s][idx][comp_ac_idx] += (q - tree.q_component_stats[s][idx][comp_ac_idx]) / tree.n_component_stats[s][idx][comp_ac_idx]
    end
    return q
end

@POMDP_require simulate(planner::JointMCTSPlanner, s, depth::Int64) begin
    mdp = planner.mdp
    P = typeof(mdp)
    @assert P <: JointMDP       # req does different thing?
    SV = statetype(P)
    @assert typeof(SV) <: AbstractVector # TODO: Is this correct?
    AV = actiontype(P)
    @assert typeof(A) <: AbstractVector
    @req discount(::P)
    @req isterminal(::P, ::SV)
    @subreq insert_node!(planner.tree, planner, s)
    @subreq estimate_value(planner.solved_estimate, mdp, s, depth)
    @req gen(::DDNOut{(:sp, :r)}, ::P, ::SV, ::A, ::typeof(planner.rng))

    # MMDP reqs
    @req get_agent_actions(::P, ::Int64)
    @req n_agents(::P)
    @req coord_graph_adj_mat(::P)

    # TODO: Should we also have this requirement for SV?
    @req isequal(::S, ::S)
    @req hash(::S)
end



function insert_node!(tree::JointMCTSTree, planner::JointMCTSPlanner, s::AbstractVector{S}) where S

    push!(tree.state_labels, s)
    tree.state_map[s] = length(tree.s_labels)

    # Now initialize the stats for new node
    n_comps = length(coord_graph_components)
    n_component_stats = Vector{Vector{Int64}}(undef, n_comps)
    q_component_stats = Vector{Vector{Float64}}(undef, n_comps)

    # TODO: Could actually make actions state-dependent if need be
    for (idx, comp) in enumerate(tree.coord_graph_components)

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
                n_component_stats[idx][comp_ac_idx] = init_N(planner.mdp, s, comp, local_action)
                q_component_stats[idx][comp_ac_idx] = init_Q(planner.mdp, s, comp, local_action)
            end
        end
    end

    # Now update tree dictionaries FOR That state
    tree.n_component_stats[s] = n_component_stats
    tree.q_component_stats[s] = q_component_stats

    # length(tree.s_labels) is just an alias for the number of state nodes
    return JointStateNode(tree, length(tree.s_labels))
end

@POMDP_require insert_node!(tree::JointMCTSTree, planner::JointMCTSPlanner, s) begin

    P = typeof(planner.mdp)
    AV = actiontype(P)
    A = eltype(AV)
    SV = typeof(s)
    S = eltype(SV)

    # TODO: Review IQ and IN
    IQ = typeof(planner.solver.init_Q)
    if !(IQ <: Number) && !(IQ <: Function)
        @req init_Q(::IQ, ::P, ::S, ::Vector{Int64}, ::AbstractVector{A})
    end

    IN = typeof(planner.solver.init_N)
    if !(IN <: Number) && !(IN <: Function)
        @req init_N(::IQ, ::P, ::S, ::Vector{Int64}, ::AbstractVector{A})
    end

    @req isequal(::S, ::S)
    @req hash(::S)
end
