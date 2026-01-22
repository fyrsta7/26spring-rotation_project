void LinearScanAllocator::AllocateRegisters() {
  DCHECK(unhandled_live_ranges().empty());
  DCHECK(active_live_ranges().empty());
  for (int reg = 0; reg < num_registers(); ++reg) {
    DCHECK(inactive_live_ranges(reg).empty());
  }

  SplitAndSpillRangesDefinedByMemoryOperand();
  data()->ResetSpillState();

  if (data()->is_trace_alloc()) {
    PrintRangeOverview();
  }

  const size_t live_ranges_size = data()->live_ranges().size();
  for (TopLevelLiveRange* range : data()->live_ranges()) {
    CHECK_EQ(live_ranges_size,
             data()->live_ranges().size());  // TODO(neis): crbug.com/831822
    if (!CanProcessRange(range)) continue;
    for (LiveRange* to_add = range; to_add != nullptr;
         to_add = to_add->next()) {
      if (!to_add->spilled()) {
        AddToUnhandled(to_add);
      }
    }
  }

  if (mode() == RegisterKind::kGeneral) {
    for (TopLevelLiveRange* current : data()->fixed_live_ranges()) {
      if (current != nullptr) {
        if (current->IsDeferredFixed()) continue;
        AddToInactive(current);
      }
    }
  } else if (mode() == RegisterKind::kDouble) {
    for (TopLevelLiveRange* current : data()->fixed_double_live_ranges()) {
      if (current != nullptr) {
        if (current->IsDeferredFixed()) continue;
        AddToInactive(current);
      }
    }
    if (kFPAliasing == AliasingKind::kCombine && check_fp_aliasing()) {
      for (TopLevelLiveRange* current : data()->fixed_float_live_ranges()) {
        if (current != nullptr) {
          if (current->IsDeferredFixed()) continue;
          AddToInactive(current);
        }
      }
      for (TopLevelLiveRange* current : data()->fixed_simd128_live_ranges()) {
        if (current != nullptr) {
          if (current->IsDeferredFixed()) continue;
          AddToInactive(current);
        }
      }
    }
  } else {
    DCHECK(mode() == RegisterKind::kSimd128);
    for (TopLevelLiveRange* current : data()->fixed_simd128_live_ranges()) {
      if (current != nullptr) {
        if (current->IsDeferredFixed()) continue;
        AddToInactive(current);
      }
    }
  }

  RpoNumber last_block = RpoNumber::FromInt(0);
  RpoNumber max_blocks =
      RpoNumber::FromInt(code()->InstructionBlockCount() - 1);
  LifetimePosition next_block_boundary =
      LifetimePosition::InstructionFromInstructionIndex(
          data()
              ->code()
              ->InstructionBlockAt(last_block)
              ->last_instruction_index())
          .NextFullStart();
  SpillMode spill_mode = SpillMode::kSpillAtDefinition;

  // Process all ranges. We also need to ensure that we have seen all block
  // boundaries. Linear scan might have assigned and spilled ranges before
  // reaching the last block and hence we would ignore control flow effects for
  // those. Not only does this produce a potentially bad assignment, it also
  // breaks with the invariant that we undo spills that happen in deferred code
  // when crossing a deferred/non-deferred boundary.
  while (!unhandled_live_ranges().empty() || last_block < max_blocks) {
    data()->tick_counter()->TickAndMaybeEnterSafepoint();
    LiveRange* current = unhandled_live_ranges().empty()
                             ? nullptr
                             : *unhandled_live_ranges().begin();
    LifetimePosition position =
        current ? current->Start() : next_block_boundary;
#ifdef DEBUG
    allocation_finger_ = position;
#endif
    // Check whether we just moved across a block boundary. This will trigger
    // for the first range that is past the current boundary.
    if (position >= next_block_boundary) {
      TRACE("Processing boundary at %d leaving %d\n",
            next_block_boundary.value(), last_block.ToInt());

      // Forward state to before block boundary
      LifetimePosition end_of_block = next_block_boundary.PrevStart().End();
      ForwardStateTo(end_of_block);

      // Remember this state.
      InstructionBlock* current_block = data()->code()->GetInstructionBlock(
          next_block_boundary.ToInstructionIndex());

      // Store current spill state (as the state at end of block). For
      // simplicity, we store the active ranges, e.g., the live ranges that
      // are not spilled.
      data()->RememberSpillState(last_block, active_live_ranges());

      // Only reset the state if this was not a direct fallthrough. Otherwise
      // control flow resolution will get confused (it does not expect changes
      // across fallthrough edges.).
      bool fallthrough =
          (current_block->PredecessorCount() == 1) &&
          current_block->predecessors()[0].IsNext(current_block->rpo_number());

      // When crossing a deferred/non-deferred boundary, we have to load or
      // remove the deferred fixed ranges from inactive.
      if ((spill_mode == SpillMode::kSpillDeferred) !=
          current_block->IsDeferred()) {
        // Update spill mode.
        spill_mode = current_block->IsDeferred()
                         ? SpillMode::kSpillDeferred
                         : SpillMode::kSpillAtDefinition;

        ForwardStateTo(next_block_boundary);

#ifdef DEBUG
        // Allow allocation at current position.
        allocation_finger_ = next_block_boundary;
#endif
        UpdateDeferredFixedRanges(spill_mode, current_block);
      }

      // Allocation relies on the fact that each non-deferred block has at
      // least one non-deferred predecessor. Check this invariant here.
      DCHECK_IMPLIES(!current_block->IsDeferred(),
                     HasNonDeferredPredecessor(current_block));

      if (!fallthrough) {
#ifdef DEBUG
        // Allow allocation at current position.
        allocation_finger_ = next_block_boundary;
#endif

        // We are currently at next_block_boundary - 1. Move the state to the
        // actual block boundary position. In particular, we have to
        // reactivate inactive ranges so that they get rescheduled for
        // allocation if they were not live at the predecessors.
        ForwardStateTo(next_block_boundary);

        RangeWithRegisterSet to_be_live(data()->allocation_zone());

        // If we end up deciding to use the state of the immediate
        // predecessor, it is better not to perform a change. It would lead to
        // the same outcome anyway.
        // This may never happen on boundaries between deferred and
        // non-deferred code, as we rely on explicit respill to ensure we
        // spill at definition.
        bool no_change_required = false;

        auto pick_state_from = [this, current_block](
                                   RpoNumber pred,
                                   RangeWithRegisterSet* to_be_live) -> bool {
          TRACE("Using information from B%d\n", pred.ToInt());
          // If this is a fall-through that is not across a deferred
          // boundary, there is nothing to do.
          bool is_noop = pred.IsNext(current_block->rpo_number());
          if (!is_noop) {
            auto& spill_state = data()->GetSpillState(pred);
            TRACE("Not a fallthrough. Adding %zu elements...\n",
                  spill_state.size());
            LifetimePosition pred_end =
                LifetimePosition::GapFromInstructionIndex(
                    this->code()->InstructionBlockAt(pred)->code_end());
            for (const auto range : spill_state) {
              // Filter out ranges that were split or had their register
              // stolen by backwards working spill heuristics. These have
              // been spilled after the fact, so ignore them.
              if (range->End() < pred_end || !range->HasRegisterAssigned())
                continue;
              to_be_live->emplace(range);
            }
          }
          return is_noop;
        };

        // Multiple cases here:
        // 1) We have a single predecessor => this is a control flow split, so
        //     just restore the predecessor state.
        // 2) We have two predecessors => this is a conditional, so break ties
        //     based on what to do based on forward uses, trying to benefit
        //     the same branch if in doubt (make one path fast).
        // 3) We have many predecessors => this is a switch. Compute union
        //     based on majority, break ties by looking forward.
        if (current_block->PredecessorCount() == 1) {
          TRACE("Single predecessor for B%d\n",
                current_block->rpo_number().ToInt());
          no_change_required =
              pick_state_from(current_block->predecessors()[0], &to_be_live);
        } else if (current_block->PredecessorCount() == 2) {
          TRACE("Two predecessors for B%d\n",
                current_block->rpo_number().ToInt());
          // If one of the branches does not contribute any information,
          // e.g. because it is deferred or a back edge, we can short cut
          // here right away.
          RpoNumber chosen_predecessor = RpoNumber::Invalid();
          if (!ConsiderBlockForControlFlow(current_block,
                                           current_block->predecessors()[0])) {
            chosen_predecessor = current_block->predecessors()[1];
          } else if (!ConsiderBlockForControlFlow(
                         current_block, current_block->predecessors()[1])) {
            chosen_predecessor = current_block->predecessors()[0];
          } else {
            chosen_predecessor = ChooseOneOfTwoPredecessorStates(
                current_block, next_block_boundary);
          }
          no_change_required = pick_state_from(chosen_predecessor, &to_be_live);

        } else {
          // Merge at the end of, e.g., a switch.
          ComputeStateFromManyPredecessors(current_block, &to_be_live);
        }

        if (!no_change_required) {
          SpillNotLiveRanges(&to_be_live, next_block_boundary, spill_mode);
          ReloadLiveRanges(to_be_live, next_block_boundary);
        }
      }
      // Update block information
      last_block = current_block->rpo_number();
      next_block_boundary = LifetimePosition::InstructionFromInstructionIndex(
                                current_block->last_instruction_index())
                                .NextFullStart();

      // We might have created new unhandled live ranges, so cycle around the
      // loop to make sure we pick the top most range in unhandled for
      // processing.
      continue;
    }

    DCHECK_NOT_NULL(current);

    TRACE("Processing interval %d:%d start=%d\n", current->TopLevel()->vreg(),
          current->relative_id(), position.value());

    // Now we can erase current, as we are sure to process it.
    unhandled_live_ranges().erase(unhandled_live_ranges().begin());

    if (current->IsTopLevel() && TryReuseSpillForPhi(current->TopLevel()))
      continue;

    ForwardStateTo(position);

    DCHECK(!current->HasRegisterAssigned() && !current->spilled());

    ProcessCurrentRange(current, spill_mode);
  }

  if (data()->is_trace_alloc()) {
    PrintRangeOverview();
  }
}
