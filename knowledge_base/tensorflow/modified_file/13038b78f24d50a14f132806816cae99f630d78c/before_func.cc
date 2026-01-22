    OptimizeNode(node_to_simplify, &nodes_to_simplify, &nodes_to_delete);
  }

  if (fetch_nodes_known_) {
    VLOG(1) << "Deleted " << nodes_to_delete.size() << " out of "
            << optimized_graph_->node_size() << " nodes.";
    EraseNodesFromGraph(nodes_to_delete, optimized_graph_);
    node_map_.reset(new NodeMap(optimized_graph_));
    BuildNodeToIdx();
  }
  return Status::OK();
}

Status DependencyOptimizer::TransitiveReduction() {
  // PRECONDITION: optimized_graph_ must be sorted topologically.
  const int num_nodes = optimized_graph_->node_size();
  // Set up a compressed version of the graph to save a constant factor in the
  // expensive algorithm below. Also cache the set of control outputs and the
  // highest index of a target of any control output from each node.
  int num_controls = 0;
  std::vector<gtl::InlinedVector<int, 4>> inputs(num_nodes);
  std::vector<gtl::InlinedVector<std::pair<int, int>, 2>> control_outputs(
      num_nodes);
  for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
    const NodeDef& node = optimized_graph_->node(node_idx);
    if (ModifiesFrameInfo(node) || !HasOpDef(node)) {
      // Ignore function nodes and nodes that modify frame info.
      continue;
    }
    for (int input_slot = 0; input_slot < node.input_size(); ++input_slot) {
      const string& input = node.input(input_slot);
      const NodeDef* input_node = node_map_->GetNode(input);
      if (ModifiesFrameInfo(*input_node) || IsMerge(*input_node)) {
        // Ignore edges from nodes that modify frame info and from Merge nodes,
        // because we cannot know which of it's input paths executes.
        continue;
      }
      const int input_node_idx = node_to_idx_[input_node];
      inputs[node_idx].push_back(input_node_idx);
      if (IsControlInput(input)) {
        ++num_controls;
        control_outputs[input_node_idx].emplace_back(node_idx, input_slot);
      }
    }
  }

  // Run the longest path in DAG algorithm for each source node that has control
  // outputs. If, for any target node of a control output, there exists a path
  // of length > 1, we can drop that control dependency.
  int num_controls_removed = 0;
  std::vector<int> longest_distance(num_nodes);
  // Map from target_index -> set of (input_slot, source_index), representing
  // the control edges to remove. We sort them in reverse order by input slot,
  // such that when we swap them out so we don't clobber the
  // node(target).input() repeated field.
  typedef std::pair<int, int> InputSlotAndSource;
  std::unordered_map<
      int, std::set<InputSlotAndSource, std::greater<InputSlotAndSource>>>
      control_edges_to_remove;
  for (int source = 0; source < num_nodes; ++source) {
    int highest_control_target = -1;
    for (const auto& control_output : control_outputs[source]) {
      if (control_output.first > highest_control_target) {
        highest_control_target = control_output.first;
      }
    }
    if (highest_control_target <= source) {
      continue;
    }
    std::fill(longest_distance.begin() + source,
              longest_distance.begin() + highest_control_target + 1, 0);
    for (int target = source + 1; target <= highest_control_target; ++target) {
      for (int input : inputs[target]) {
        // If the input node is before source in the topo order, no path
        // source -> input -> target can exits and we can skip it.
        // Also only extend a path from the source itself or from nodes that
        // have a path from source, indicated by longest_distance[input] > 0.
        if (input == source ||
            (input > source && longest_distance[input] > 0)) {
          // If source -> input -> target is longer than the longest
          // path so far from source -> target, update the longest_distance.
          int candidate_longest_distance = longest_distance[input] + 1;
          if (candidate_longest_distance > longest_distance[target]) {
            longest_distance[target] = candidate_longest_distance;
          }
        }
      }
    }

    // If the longest path from source to target of a control dependency is
    // longer than 1, there exists an alternate path, and we can eliminate the
    // redundant direct control dependency.
    for (const auto& control_output : control_outputs[source]) {
      const int target = control_output.first;
      if (longest_distance[target] > 1) {
        const int input_slot = control_output.second;
        control_edges_to_remove[target].emplace(input_slot, source);
      }
    }
  }

  for (const auto& it : control_edges_to_remove) {
    const int target = it.first;
    NodeDef* target_node = optimized_graph_->mutable_node(target);
    for (const InputSlotAndSource& slot_and_source : it.second) {
      const int input_slot = slot_and_source.first;
      const int source = slot_and_source.second;
