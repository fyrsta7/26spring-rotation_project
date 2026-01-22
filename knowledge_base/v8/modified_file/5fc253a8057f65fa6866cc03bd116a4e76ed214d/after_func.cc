void SplinterRangesInDeferredBlocks(RegisterAllocationData *data) {
  InstructionSequence *code = data->code();
  int code_block_count = code->InstructionBlockCount();
  Zone *zone = data->allocation_zone();
  ZoneVector<BitVector *> &in_sets = data->live_in_sets();

  for (int i = 0; i < code_block_count; ++i) {
    InstructionBlock *block = code->InstructionBlockAt(RpoNumber::FromInt(i));
    if (!block->IsDeferred()) continue;

    RpoNumber last_deferred = block->last_deferred();
    // last_deferred + 1 is not deferred, so no point in visiting it.
    i = last_deferred.ToInt() + 1;

    LifetimePosition first_cut = LifetimePosition::GapFromInstructionIndex(
        block->first_instruction_index());

    LifetimePosition last_cut = LifetimePosition::GapFromInstructionIndex(
        static_cast<int>(code->instructions().size()));

    const BitVector *in_set = in_sets[block->rpo_number().ToInt()];
    BitVector ranges_to_splinter(*in_set, zone);
    InstructionBlock *last = code->InstructionBlockAt(last_deferred);
    for (int deferred_id = block->rpo_number().ToInt();
         deferred_id <= last->rpo_number().ToInt(); ++deferred_id) {
      const BitVector *ins = in_sets[deferred_id];
      ranges_to_splinter.Union(*ins);
      const BitVector *outs = LiveRangeBuilder::ComputeLiveOut(
          code->InstructionBlockAt(RpoNumber::FromInt(deferred_id)), data);
      ranges_to_splinter.Union(*outs);
    }

    int last_index = last->last_instruction_index();
    if (code->InstructionAt(last_index)->opcode() ==
        ArchOpcode::kArchDeoptimize) {
      ++last_index;
    }
    last_cut = LifetimePosition::GapFromInstructionIndex(last_index);

    BitVector::Iterator iterator(&ranges_to_splinter);

    while (!iterator.Done()) {
      int range_id = iterator.Current();
      iterator.Advance();

      TopLevelLiveRange *range = data->live_ranges()[range_id];
      CreateSplinter(range, data, first_cut, last_cut);
    }
  }
}
