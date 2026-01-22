Reduction JSInliningHeuristic::Reduce(Node* node) {
  if (!IrOpcode::IsInlineeOpcode(node->opcode())) return NoChange();

  // Check if we already saw that {node} before, and if so, just skip it.
  if (seen_.find(node->id()) != seen_.end()) return NoChange();
  seen_.insert(node->id());

  // Check if the {node} is an appropriate candidate for inlining.
  Node* callee = node->InputAt(0);
  Candidate candidate;
  candidate.node = node;
  candidate.num_functions = CollectFunctions(
      callee, candidate.functions, kMaxCallPolymorphism, candidate.shared_info);
  if (candidate.num_functions == 0) {
    return NoChange();
  } else if (candidate.num_functions > 1 && !FLAG_polymorphic_inlining) {
    TRACE(
        "Not considering call site #%d:%s, because polymorphic inlining "
        "is disabled\n",
        node->id(), node->op()->mnemonic());
    return NoChange();
  }

  // Functions marked with %SetForceInlineFlag are immediately inlined.
  bool can_inline = false, force_inline = true, small_inline = true;
  candidate.total_size = 0;
  Node* frame_state = NodeProperties::GetFrameStateInput(node);
  FrameStateInfo const& frame_info = OpParameter<FrameStateInfo>(frame_state);
  Handle<SharedFunctionInfo> frame_shared_info;
  for (int i = 0; i < candidate.num_functions; ++i) {
    Handle<SharedFunctionInfo> shared =
        candidate.functions[i].is_null()
            ? candidate.shared_info
            : handle(candidate.functions[i]->shared());
    if (!shared->force_inline()) {
      force_inline = false;
    }
    candidate.can_inline_function[i] = CanInlineFunction(shared);
    // Do not allow direct recursion i.e. f() -> f(). We still allow indirect
    // recurion like f() -> g() -> f(). The indirect recursion is helpful in
    // cases where f() is a small dispatch function that calls the appropriate
    // function. In the case of direct recursion, we only have some static
    // information for the first level of inlining and it may not be that useful
    // to just inline one level in recursive calls. In some cases like tail
    // recursion we may benefit from recursive inlining, if we have additional
    // analysis that converts them to iterative implementations. Though it is
    // not obvious if such an anlysis is needed.
    if (frame_info.shared_info().ToHandle(&frame_shared_info) &&
        *frame_shared_info == *shared) {
      TRACE("Not considering call site #%d:%s, because of recursive inlining\n",
            node->id(), node->op()->mnemonic());
      candidate.can_inline_function[i] = false;
    }
    if (candidate.can_inline_function[i]) {
      can_inline = true;
      candidate.total_size += shared->bytecode_array()->length();
    }
    if (!IsSmallInlineFunction(shared)) {
      small_inline = false;
    }
  }
  if (force_inline) return InlineCandidate(candidate, true);
  if (!can_inline) return NoChange();

  // Stop inlining once the maximum allowed level is reached.
  int level = 0;
  for (Node* frame_state = NodeProperties::GetFrameStateInput(node);
       frame_state->opcode() == IrOpcode::kFrameState;
       frame_state = NodeProperties::GetFrameStateInput(frame_state)) {
    FrameStateInfo const& frame_info = OpParameter<FrameStateInfo>(frame_state);
    if (FrameStateFunctionInfo::IsJSFunctionType(frame_info.type())) {
      if (++level > FLAG_max_inlining_levels) {
        TRACE(
            "Not considering call site #%d:%s, because inlining depth "
            "%d exceeds maximum allowed level %d\n",
            node->id(), node->op()->mnemonic(), level,
            FLAG_max_inlining_levels);
        return NoChange();
      }
    }
  }

  // Gather feedback on how often this call site has been hit before.
  if (node->opcode() == IrOpcode::kJSCall) {
    CallParameters const p = CallParametersOf(node->op());
    candidate.frequency = p.frequency();
  } else {
    ConstructParameters const p = ConstructParametersOf(node->op());
    candidate.frequency = p.frequency();
  }

  // Handling of special inlining modes right away:
  //  - For restricted inlining: stop all handling at this point.
  //  - For stressing inlining: immediately handle all functions.
  switch (mode_) {
    case kRestrictedInlining:
      return NoChange();
    case kStressInlining:
      return InlineCandidate(candidate, false);
    case kGeneralInlining:
      break;
  }

  // Don't consider a {candidate} whose frequency is below the
  // threshold, i.e. a call site that is only hit once every N
  // invocations of the caller.
  if (candidate.frequency.IsKnown() &&
      candidate.frequency.value() < FLAG_min_inlining_frequency) {
    return NoChange();
  }

  // Forcibly inline small functions here. In the case of polymorphic inlining
  // small_inline is set only when all functions are small.
  if (small_inline &&
      cumulative_count_ <= FLAG_max_inlined_bytecode_size_absolute) {
    TRACE("Inlining small function(s) at call site #%d:%s\n", node->id(),
          node->op()->mnemonic());
    return InlineCandidate(candidate, true);
  }

  // In the general case we remember the candidate for later.
  candidates_.insert(candidate);
  return NoChange();
}
