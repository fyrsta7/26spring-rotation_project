bool LinearScanAllocator::TryReuseSpillForPhi(TopLevelLiveRange* range) {
  if (!range->is_phi()) return false;

  DCHECK(!range->HasSpillOperand());
  RegisterAllocationData::PhiMapValue* phi_map_value =
      data()->GetPhiMapValueFor(range);
  const PhiInstruction* phi = phi_map_value->phi();
  const InstructionBlock* block = phi_map_value->block();
  // Count the number of spilled operands.
  size_t spilled_count = 0;
  LiveRange* first_op = nullptr;
  for (size_t i = 0; i < phi->operands().size(); i++) {
    int op = phi->operands()[i];
    LiveRange* op_range = data()->GetOrCreateLiveRangeFor(op);
    if (!op_range->TopLevel()->HasSpillRange()) continue;
    const InstructionBlock* pred =
        code()->InstructionBlockAt(block->predecessors()[i]);
    LifetimePosition pred_end =
        LifetimePosition::InstructionFromInstructionIndex(
            pred->last_instruction_index());
    while (op_range != nullptr && !op_range->CanCover(pred_end)) {
      op_range = op_range->next();
    }
    if (op_range != nullptr && op_range->spilled()) {
      spilled_count++;
      if (first_op == nullptr) {
        first_op = op_range->TopLevel();
      }
    }
  }

  // Only continue if more than half of the operands are spilled.
  if (spilled_count * 2 <= phi->operands().size()) {
    return false;
  }

  // Try to merge the spilled operands and count the number of merged spilled
  // operands.
  DCHECK_NOT_NULL(first_op);
  SpillRange* first_op_spill = first_op->TopLevel()->GetSpillRange();
  size_t num_merged = 1;
  for (size_t i = 1; i < phi->operands().size(); i++) {
    int op = phi->operands()[i];
    TopLevelLiveRange* op_range = data()->live_ranges()[op];
    if (!op_range->HasSpillRange()) continue;
    SpillRange* op_spill = op_range->GetSpillRange();
    if (op_spill == first_op_spill || first_op_spill->TryMerge(op_spill)) {
      num_merged++;
    }
  }

  // Only continue if enough operands could be merged to the
  // same spill slot.
  if (num_merged * 2 <= phi->operands().size() ||
      AreUseIntervalsIntersecting(first_op_spill->interval(),
                                  range->first_interval())) {
    return false;
  }

  // If the range does not need register soon, spill it to the merged
  // spill range.
  LifetimePosition next_pos = range->Start();
  if (next_pos.IsGapPosition()) next_pos = next_pos.NextStart();
  UsePosition* pos = range->NextUsePositionRegisterIsBeneficial(next_pos);
  if (pos == nullptr) {
    SpillRange* spill_range =
        range->TopLevel()->HasSpillRange()
            ? range->TopLevel()->GetSpillRange()
            : data()->AssignSpillRangeToLiveRange(range->TopLevel());
    bool merged = first_op_spill->TryMerge(spill_range);
    if (!merged) return false;
    Spill(range);
    return true;
  } else if (pos->pos() > range->Start().NextStart()) {
    SpillRange* spill_range =
        range->TopLevel()->HasSpillRange()
            ? range->TopLevel()->GetSpillRange()
            : data()->AssignSpillRangeToLiveRange(range->TopLevel());
    bool merged = first_op_spill->TryMerge(spill_range);
    if (!merged) return false;
    SpillBetween(range, range->Start(), pos->pos());
    return true;
  }
  return false;
}
