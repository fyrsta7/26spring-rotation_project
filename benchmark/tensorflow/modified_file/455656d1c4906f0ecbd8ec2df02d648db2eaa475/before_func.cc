StatusOr<bool> HloCSE::Run(HloModule* module) {
  bool changed = false;
  const std::function<bool(const HloInstruction*, const HloInstruction*)>
      eq_instructions = std::equal_to<const HloInstruction*>();
  const std::function<bool(const HloComputation*, const HloComputation*)>
      eq_computations = [](const HloComputation* lhs,
                           const HloComputation* rhs) { return *lhs == *rhs; };

  auto cse_equal = [&](const HloInstruction* lhs, const HloInstruction* rhs) {
    return lhs->Identical(*rhs, eq_instructions, eq_computations,
                          is_layout_sensitive_);
  };

  for (auto* computation : module->computations()) {
    if (only_fusion_computations_ && !computation->IsFusionComputation()) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(bool combined,
                        CombineConstants(computation, is_layout_sensitive_));
    changed |= combined;

    // HLO instructions are grouped into equivalency classes by using the
    // cse_equal predicate defined above. This set holds a representative
    // instruction for each class.
    absl::flat_hash_set<HloInstruction*, decltype(&CseHash),
                        decltype(cse_equal)>
        representatives(/*N=*/computation->instruction_count() + 1, &CseHash,
                        cse_equal);
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // If the instruction has zero operands (constants, parameters, etc.) skip
      // over it.
      if (instruction->operand_count() == 0 &&
          instruction->opcode() != HloOpcode::kPartitionId &&
          instruction->opcode() != HloOpcode::kReplicaId) {
        continue;
      }
      // Skip instructions which have side effects.
      if (instruction->HasSideEffect()) {
        continue;
      }

      auto it = representatives.find(instruction);
      if (it != representatives.end()) {
        HloInstruction* equivalent_instruction = *it;
        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWith(equivalent_instruction));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        changed = true;
        continue;
      }
      representatives.insert(instruction);
    }
  }
  return changed;
}