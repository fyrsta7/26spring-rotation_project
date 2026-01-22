  HloInstruction* new_a = a->AddInstruction(HloInstruction::CreateSlice(
      a->shape(), new_dot, start_indices, limit_indices, strides));
  TF_RETURN_IF_ERROR(a->ReplaceAllUsesWith(new_a));

  start_indices[slice_dim] = limit_indices[slice_dim];
  limit_indices[slice_dim] = new_dot_shape.dimensions(slice_dim);
  HloInstruction* new_b = b->AddInstruction(HloInstruction::CreateSlice(
      b->shape(), new_dot, start_indices, limit_indices, strides));
  TF_RETURN_IF_ERROR(b->ReplaceAllUsesWith(new_b));

  return new_dot;
}

StatusOr<bool> MergeDots(HloComputation* comp, int64_t max_size_to_merge) {
  auto is_merge_candidate = [&](HloInstruction* instr) {
    int64_t bytes = ShapeUtil::ByteSizeOfElements(instr->shape());
    for (const HloInstruction* operand : instr->operands()) {
      bytes += ShapeUtil::ByteSizeOfElements(operand->shape());
    }
    return bytes <= max_size_to_merge;
  };

  // Collect equivalence classes.  Specifically, create the map
  //
  //   instruction -> [canonical dots that use the instruction].
  //
  // We'll then try to merge dots within each equivalence class.  A dot will be
  // a member of two equivalence classes (because it has two operands), but if
  // it's merged with a dot from one equivalence class, it won't also be merged
  // in another class.
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloInstruction*>>
      equivalence_classes;
  for (HloInstruction* instr : comp->instructions()) {
    // Cowardly skip instructions with control dependencies.
    if (!IsCanonicalDot(instr) || !instr->control_predecessors().empty() ||
        !instr->control_successors().empty()) {
      continue;
    }
    for (HloInstruction* operand : instr->operands()) {
      equivalence_classes[operand].insert(instr);
    }
  }

  // Remove "uninteresting" equivalence classes where either
  //
  //  - there's just one instruction (nothing to merge!), or
  //  - there are zero instructions marked as mergeable.  (Our contract is that
  //    at least one instruction of the pair needs to be mergeable in order for
  //    us to merge.)
  absl::erase_if(
      equivalence_classes,
      [&](const std::pair<const HloInstruction*,
                          absl::flat_hash_set<HloInstruction*>>& kv) {
        const auto& v = kv.second;
        return v.size() < 2 || absl::c_none_of(v, is_merge_candidate);
      });

  // Are there any possible optimization opportunities?
  if (equivalence_classes.empty()) {
    return false;
  }

  // Build a dependency graph representing the whole computation.
  tensorflow::GraphCycles graph;

  absl::flat_hash_map<HloInstruction*, int32_t> graph_ids_map;
  auto graph_id = [&](HloInstruction* instr) {
    auto it_and_inserted = graph_ids_map.emplace(instr, -1);
    auto it = it_and_inserted.first;
    auto inserted = it_and_inserted.second;
    if (inserted) {
      it->second = graph.NewNode();
    }
    return it->second;
  };

  // Iteration order doesn't matter for correctness, but graph.InsertEdge() is
  // *much* faster if we iterate in topological order.
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    int32_t id = graph_id(instr);
    for (HloInstruction* operand : instr->operands()) {
      CHECK(graph.InsertEdge(graph_id(operand), id));
    }
    for (HloInstruction* control_pred : instr->control_predecessors()) {
      CHECK(graph.InsertEdge(graph_id(control_pred), id));
    }
  }

  // Merge within equivalence classes.  We keep a set of all instructions that
  // have been merged so we don't try to merge an instruction twice.  We'll
  // remove these dead instructions at the end of the pass.  (We can't remove
  // them earlier because removing an instruction deletes it; we'd then have
  // dangling pointers in our hashtable!)
  absl::flat_hash_set<HloInstruction*> dead_instrs;
  for (auto& kv : equivalence_classes) {
    // For determinism, iterate in order of the instructions' IDs.
    absl::InlinedVector<HloInstruction*, 16> dots(kv.second.begin(),
                                                  kv.second.end());
    absl::c_sort(dots, [](const HloInstruction* a, const HloInstruction* b) {
      return a->unique_id() < b->unique_id();
    });

    // Try merging all pairs of dots in this equivalence class.
    for (int64_t i = 0; i < dots.size(); i++) {
      HloInstruction*& a = dots[i];
      if (a == nullptr) {
        continue;
      }
      for (int64_t j = i + 1; j < dots.size(); j++) {
        HloInstruction* b = dots[j];
        if (b == nullptr) {
          continue;
        }
        int32_t a_id = graph_id(a);
        int32_t b_id = graph_id(b);

        if (dead_instrs.contains(a) || dead_instrs.contains(b) ||
            graph.IsReachableNonConst(a_id, b_id) ||
            graph.IsReachableNonConst(b_id, a_id) ||
            (!is_merge_candidate(a) && !is_merge_candidate(b))) {
          continue;
        }

        TF_ASSIGN_OR_RETURN(HloInstruction * merged, TryMergeSameOperand(a, b));
        if (merged != nullptr) {
          int32_t merged_id = graph_id(merged);
          graph.InsertEdge(a_id, merged_id);
          graph.InsertEdge(b_id, merged_id);
          for (int32_t succ : graph.SuccessorsCopy(a_id)) {
            graph.InsertEdge(merged_id, succ);
          }
          for (int32_t succ : graph.SuccessorsCopy(b_id)) {
            graph.InsertEdge(merged_id, succ);
          }

          dead_instrs.insert(a);
          dead_instrs.insert(b);
          dots[i] = merged;
