void LinearScanAllocator::ComputeStateFromManyPredecessors(
    InstructionBlock* current_block, RangeWithRegisterSet* to_be_live) {
  struct Vote {
    size_t count;
    int used_registers[RegisterConfiguration::kMaxRegisters];
  };
  struct TopLevelLiveRangeComparator {
    bool operator()(const TopLevelLiveRange* lhs,
                    const TopLevelLiveRange* rhs) const {
      return lhs->vreg() < rhs->vreg();
    }
  };
  ZoneMap<TopLevelLiveRange*, Vote, TopLevelLiveRangeComparator> counts(
      data()->allocation_zone());
  int deferred_blocks = 0;
  for (RpoNumber pred : current_block->predecessors()) {
    if (!ConsiderBlockForControlFlow(current_block, pred)) {
      // Back edges of a loop count as deferred here too.
      deferred_blocks++;
      continue;
    }
    const auto& pred_state = data()->GetSpillState(pred);
    for (LiveRange* range : pred_state) {
      // We might have spilled the register backwards, so the range we
      // stored might have lost its register. Ignore those.
      if (!range->HasRegisterAssigned()) continue;
      TopLevelLiveRange* toplevel = range->TopLevel();
      auto previous = counts.find(toplevel);
      if (previous == counts.end()) {
        auto result = counts.emplace(std::make_pair(toplevel, Vote{1, {0}}));
        CHECK(result.second);
        result.first->second.used_registers[range->assigned_register()]++;
      } else {
        previous->second.count++;
        previous->second.used_registers[range->assigned_register()]++;
      }
    }
  }

  // Choose the live ranges from the majority.
  const size_t majority =
      (current_block->PredecessorCount() + 2 - deferred_blocks) / 2;
  bool taken_registers[RegisterConfiguration::kMaxRegisters] = {false};
  auto assign_to_live = [this, counts, majority](
                            std::function<bool(TopLevelLiveRange*)> filter,
                            RangeWithRegisterSet* to_be_live,
                            bool* taken_registers) {
    bool check_aliasing =
        kFPAliasing == AliasingKind::kCombine && check_fp_aliasing();
    for (const auto& val : counts) {
      if (!filter(val.first)) continue;
      if (val.second.count >= majority) {
        int register_max = 0;
        int reg = kUnassignedRegister;
        bool conflict = false;
        int num_regs = num_registers();
        int num_codes = num_allocatable_registers();
        const int* codes = allocatable_register_codes();
        MachineRepresentation rep = val.first->representation();
        if (check_aliasing && (rep == MachineRepresentation::kFloat32 ||
                               rep == MachineRepresentation::kSimd128 ||
                               rep == MachineRepresentation::kSimd256))
          GetFPRegisterSet(rep, &num_regs, &num_codes, &codes);
        for (int idx = 0; idx < num_regs; idx++) {
          int uses = val.second.used_registers[idx];
          if (uses == 0) continue;
          if (uses > register_max || (conflict && uses == register_max)) {
            reg = idx;
            register_max = uses;
            conflict = check_aliasing ? CheckConflict(rep, reg, to_be_live)
                                      : taken_registers[reg];
          }
        }
        if (conflict) {
          reg = kUnassignedRegister;
        } else if (!check_aliasing) {
          taken_registers[reg] = true;
        }
        to_be_live->emplace(val.first, reg);
        TRACE("Reset %d as live due vote %zu in %s\n",
              val.first->TopLevel()->vreg(), val.second.count,
              RegisterName(reg));
      }
    }
  };
  // First round, process fixed registers, as these have precedence.
  // There is only one fixed range per register, so we cannot have
  // conflicts.
  assign_to_live([](TopLevelLiveRange* r) { return r->IsFixed(); }, to_be_live,
                 taken_registers);
  // Second round, process the rest.
  assign_to_live([](TopLevelLiveRange* r) { return !r->IsFixed(); }, to_be_live,
                 taken_registers);
}
