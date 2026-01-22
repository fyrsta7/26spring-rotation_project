void LiveRangeBuilder::ProcessPhis(const InstructionBlock* block,
                                   BitVector* live) {
  for (PhiInstruction* phi : block->phis()) {
    // The live range interval already ends at the first instruction of the
    // block.
    int phi_vreg = phi->virtual_register();
    live->Remove(phi_vreg);
    InstructionOperand* hint = nullptr;
    Instruction* instr = GetLastInstruction(
        code(), code()->InstructionBlockAt(block->predecessors()[0]));
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
