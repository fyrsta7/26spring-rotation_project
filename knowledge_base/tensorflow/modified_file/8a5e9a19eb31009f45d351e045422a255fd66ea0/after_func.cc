           // parent GTE and its children below.
           if (inst->opcode() == HloOpcode::kBitcast &&
               inst->operand(0)->opcode() == HloOpcode::kGetTupleElement &&
               inst->operand(0)->operand(0)->opcode() ==
                   HloOpcode::kParameter) {
             return true;
           }

           return inst->opcode() == HloOpcode::kGetTupleElement &&
                  !LooksLikeAnActivation(inst);
         });
}

absl::optional<MemorySpaceAssignment::BufferInterval>
FindCrossProgramPrefetchCandidate(
    const HloAliasAnalysis& alias_analysis, const HloLiveRange& hlo_live_range,
    const MemorySpaceAssignment::Options& options) {
  std::vector<MemorySpaceAssignment::BufferInterval> candidates;
  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    CHECK_GE(buffer.values().size(), 1);
    const HloValue* value = buffer.values().at(0);
    if (IsCrossProgramPrefetchCandidate(*value, options)) {
      MemorySpaceAssignment::BufferInterval interval;
      interval.buffer = value;
      interval.size = options.size_fn(*value);
      interval.start = 0;
      interval.end = hlo_live_range.schedule_end_time();
      interval.need_allocation = true;
      interval.colocations = {++buffer.values().begin(), buffer.values().end()};
      candidates.emplace_back(interval);
    }
  }

  // The buffer_interval_compare ought to do a good job picking the most
  // appropriate buffer to cross program prefetch, but empirically, it makes
  // worse choices than just picking the largest buffer.
  // TODO(b/152421603): Investigate.
  auto size_compare = [](const auto& x, const auto& y) {
    if (x.size == y.size) {
      // When both buffers are of same size, we prefer the one that is used to
      // produce larger tensors in its consumer instructions.
      auto get_use_size =
          [](const MemorySpaceAssignment::BufferInterval& bi) -> int64 {
        int64 use_size = 0;
        for (const auto& use : bi.buffer->uses()) {
          use_size += ShapeUtil::ElementsInRecursive(use.instruction->shape());
        }
        return use_size;
      };
      return get_use_size(x) < get_use_size(y);
    }
