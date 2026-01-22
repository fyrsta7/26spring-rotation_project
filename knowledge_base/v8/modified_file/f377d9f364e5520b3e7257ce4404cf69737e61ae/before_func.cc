void RegisterAllocatorVerifier::ValidatePendingAssessment(
    RpoNumber block_id, InstructionOperand op,
    BlockAssessments* current_assessments, const PendingAssessment* assessment,
    int virtual_register) {
  // When validating a pending assessment, it is possible some of the
  // assessments
  // for the original operand (the one where the assessment was created for
  // first) are also pending. To avoid recursion, we use a work list. To
  // deal with cycles, we keep a set of seen nodes.
  ZoneQueue<std::pair<const PendingAssessment*, int>> worklist(zone());
  ZoneSet<RpoNumber> seen(zone());
  worklist.push(std::make_pair(assessment, virtual_register));
  seen.insert(block_id);

  while (!worklist.empty()) {
    auto work = worklist.front();
    const PendingAssessment* current_assessment = work.first;
    int current_virtual_register = work.second;
    InstructionOperand current_operand = current_assessment->operand();
    worklist.pop();

    const InstructionBlock* origin = current_assessment->origin();
    CHECK(origin->PredecessorCount() > 1 || origin->phis().size() > 0);

    // Check if the virtual register is a phi first, instead of relying on
    // the incoming assessments. In particular, this handles the case
    // v1 = phi v0 v0, which structurally is identical to v0 having been
    // defined at the top of a diamond, and arriving at the node joining the
    // diamond's branches.
    const PhiInstruction* phi = nullptr;
    for (const PhiInstruction* candidate : origin->phis()) {
      if (candidate->virtual_register() == current_virtual_register) {
        phi = candidate;
        break;
      }
    }

    int op_index = 0;
    for (RpoNumber pred : origin->predecessors()) {
      int expected =
          phi != nullptr ? phi->operands()[op_index] : current_virtual_register;

      ++op_index;
      auto pred_assignment = assessments_.find(pred);
      if (pred_assignment == assessments_.end()) {
        CHECK(origin->IsLoopHeader());
        auto todo_iter = outstanding_assessments_.find(pred);
        DelayedAssessments* set = nullptr;
        if (todo_iter == outstanding_assessments_.end()) {
          set = new (zone()) DelayedAssessments(zone());
          outstanding_assessments_.insert(std::make_pair(pred, set));
        } else {
          set = todo_iter->second;
        }
        set->AddDelayedAssessment(current_operand, expected);
        continue;
      }

      const BlockAssessments* pred_assessments = pred_assignment->second;
      auto found_contribution = pred_assessments->map().find(current_operand);
      CHECK(found_contribution != pred_assessments->map().end());
      Assessment* contribution = found_contribution->second;

      switch (contribution->kind()) {
        case Final:
          ValidateFinalAssessment(
              block_id, current_operand, current_assessments,
              FinalAssessment::cast(contribution), expected);
          break;
        case Pending: {
          // This happens if we have a diamond feeding into another one, and
          // the inner one never being used - other than for carrying the value.
          const PendingAssessment* next = PendingAssessment::cast(contribution);
          if (seen.find(pred) == seen.end()) {
            worklist.push({next, expected});
            seen.insert(pred);
          }
          // Note that we do not want to finalize pending assessments at the
          // beginning of a block - which is the information we'd have
          // available here. This is because this operand may be reused to
          // define
          // duplicate phis.
          break;
        }
      }
    }
  }
  // If everything checks out, we may make the assessment.
  current_assessments->map()[op] =
      new (zone()) FinalAssessment(virtual_register, assessment);
}
