static bool IsOrContainsIllegalInstr(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kAfterAll ||
      instr->opcode() == HloOpcode::kRng) {
    return true;
  }
  for (const HloComputation* c : instr->called_computations()) {
    if (absl::c_any_of(c->instructions(), IsOrContainsIllegalInstr)) {
      return true;
    }
  }
  return false;
}

StatusOr<bool> HloConstantFolding::Run(HloModule* module) {
  // Limit the constant folding to 0 iterations to skip folding loops. This
  // retains the behavior from before while loop support in HloEvaluator and may
  // be revised.
  auto evaluator = absl::make_unique<HloEvaluator>(/*max_loop_iterations=*/0);

  XLA_VLOG_LINES(2,
                 "HloConstantFolding::Run(), before:\n" + module->ToString());
  bool changed = false;

  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      // Skip dead code.
      if (instruction->IsDead()) {
        continue;
      }

      // We only handle instructions where
      //
      //  - at least one operand is a constant, and
      //  - all other operands are either constants or broadcast(constant).
      //
      // Why this particular set of rules around broadcasts?
      //
      //  - We don't want to fold broadcast(constant) on its own, because in
      //    general it's "simpler" to remember that it's a broadcast.  Also,
      //    algsimp will fold an all-one-value constant into a broadcast, so
      //    we'd just end up fighting with it.
      //
      //  - We don't want to fold an op where all operands are broadcasts of
      //    constants, because algsimp will transform op(broadcast(constant) =>
      //    broadcast(op(constant)).  Then we can constant-fold the smaller op.
      //
      //  - So the only remaining case is where some but not all operands are
      //    broadcasts of constants, e.g. op(constant, broadcast(constant)).
      //
      if (!absl::c_any_of(instruction->operands(),
                          [](const HloInstruction* operand) {
                            return operand->opcode() == HloOpcode::kConstant;
                          }) ||
          !absl::c_all_of(
              instruction->operands(), [](const HloInstruction* operand) {
                return operand->opcode() == HloOpcode::kConstant ||
                       (operand->opcode() == HloOpcode::kBroadcast &&
                        operand->operand(0)->opcode() == HloOpcode::kConstant);
              })) {
        continue;
      }

      // Don't fold Constant, Parameter, and Tuple instructions.  Tuple
      // constants are not directly supported by any backends, hence folding
      // Tuple is not useful and would in fact be expanded back into kTuple by
      // Algebraic Simplifier.
      //
      // (We do allow folding subcomputations that contain these instructions.)
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->opcode() == HloOpcode::kConstant ||
          instruction->opcode() == HloOpcode::kTuple) {
        continue;
      }

      // Broadcasts dramatically increase the size of constants, which is often
      // detrimental to performance and memory capacity, so do not fold
      // broadcasts.
      if (instruction->opcode() == HloOpcode::kBroadcast ||
          instruction->opcode() == HloOpcode::kIota) {
        continue;
      }

      // Check for instructions that we can't fold even if they appear inside of
      // a subcomputation (e.g. a kCall).
      if (IsOrContainsIllegalInstr(instruction)) {
        continue;
      }

      // Don't constant-fold side-effecting instructions or instructions which
      // contain side-effecting instructions.
      if (instruction->HasSideEffect()) {
        continue;
      }

      // Don't constant fold unless it's a net positive or the output is small.
      if (instruction->shape().IsArray()) {
        int64_t elements_in_removed_operands = 0;
        for (HloInstruction* operand : instruction->operands()) {
          if (operand->user_count() == 1 && operand->shape().IsArray()) {
            elements_in_removed_operands +=
                ShapeUtil::ElementsIn(operand->shape());
          }
        }
        int64_t elements_in_constant =
            ShapeUtil::ElementsIn(instruction->shape());

        static const int64_t kMaximumConstantSizeElements = 45 * 1000 * 1000;
        if (elements_in_constant > elements_in_removed_operands &&
            elements_in_constant > kMaximumConstantSizeElements) {
          continue;
        }
      }

      Literal result;
      // Currently we skip unimplemented operations.
      // TODO(b/35975797): Fold constant computations for more operations.
      if (!evaluator->TryEvaluate(
              instruction, &result,
              /*recursively_evaluate_nonconstant_operands=*/true)) {
        VLOG(2) << "Constant folding failed for instruction: "
