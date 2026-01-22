void LiveRangeBuilder::ProcessPhis(const InstructionBlock* block,
                                   BitVector* live) {
  for (PhiInstruction* phi : block->phis()) {
    // The live range interval already ends at the first instruction of the
    // block.
    int phi_vreg = phi->virtual_register();
    live->Remove(phi_vreg);
    InstructionOperand* hint = nullptr;
    const InstructionBlock::Predecessors& predecessors = block->predecessors();
    const InstructionBlock* predecessor_block =
        code()->InstructionBlockAt(predecessors[0]);
    const Instruction* instr = GetLastInstruction(code(), predecessor_block);
    if (predecessor_block->IsDeferred()) {
      // "Prefer the hint from the first non-deferred predecessor, if any.
      for (size_t i = 1; i < predecessors.size(); ++i) {
        predecessor_block = code()->InstructionBlockAt(predecessors[i]);
        if (!predecessor_block->IsDeferred()) {
          instr = GetLastInstruction(code(), predecessor_block);
          break;
        }
      }
    }
    DCHECK_NOT_NULL(instr);

    for (MoveOperands* move : *instr->GetParallelMove(Instruction::END)) {
      InstructionOperand& to = move->destination();
      if (to.IsUnallocated() &&
          UnallocatedOperand::cast(to).virtual_register() == phi_vreg) {
        hint = &move->source();
        break;
      }
    }
    DCHECK(hint != nullptr);
    LifetimePosition block_start = LifetimePosition::GapFromInstructionIndex(
        block->first_instruction_index());
    UsePosition* use_pos = Define(block_start, &phi->output(), hint,
                                  UsePosition::HintTypeForOperand(*hint));
    MapPhiHint(hint, use_pos);
  }
}
