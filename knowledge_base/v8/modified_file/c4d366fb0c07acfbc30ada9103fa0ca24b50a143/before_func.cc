Node* AstGraphBuilder::TryLoadDynamicVariable(
    Variable* variable, Handle<String> name, BailoutId bailout_id,
    FrameStateBeforeAndAfter& states, const VectorSlotPair& feedback,
    OutputFrameStateCombine combine, TypeofMode typeof_mode) {
  VariableMode mode = variable->mode();

  if (mode == DYNAMIC_GLOBAL) {
    uint32_t bitset = ComputeBitsetForDynamicGlobal(variable);
    if (bitset == kFullCheckRequired) return nullptr;

    // We are using two blocks to model fast and slow cases.
    BlockBuilder fast_block(this);
    BlockBuilder slow_block(this);
    environment()->Push(jsgraph()->TheHoleConstant());
    slow_block.BeginBlock();
    environment()->Pop();
    fast_block.BeginBlock();

    // Perform checks whether the fast mode applies, by looking for any
    // extension object which might shadow the optimistic declaration.
    for (int depth = 0; bitset != 0; bitset >>= 1, depth++) {
      if ((bitset & 1) == 0) continue;
      Node* load = NewNode(
          javascript()->LoadContext(depth, Context::EXTENSION_INDEX, false),
          current_context());
      Node* check = NewNode(javascript()->StrictEqual(), load,
                            jsgraph()->TheHoleConstant());
      fast_block.BreakUnless(check, BranchHint::kTrue);
    }

    // Fast case, because variable is not shadowed. Perform global slot load.
    Node* fast = BuildGlobalLoad(name, feedback, typeof_mode);
    states.AddToNode(fast, bailout_id, combine);
    environment()->Push(fast);
    slow_block.Break();
    environment()->Pop();
    fast_block.EndBlock();

    // Slow case, because variable potentially shadowed. Perform dynamic lookup.
    const Operator* op = javascript()->LoadDynamic(name, typeof_mode);
    Node* slow = NewNode(op, BuildLoadFeedbackVector(), current_context());
    states.AddToNode(slow, bailout_id, combine);
    environment()->Push(slow);
    slow_block.EndBlock();

    return environment()->Pop();
  }

  if (mode == DYNAMIC_LOCAL) {
    uint32_t bitset = ComputeBitsetForDynamicContext(variable);
    if (bitset == kFullCheckRequired) return nullptr;

    // We are using two blocks to model fast and slow cases.
    BlockBuilder fast_block(this);
    BlockBuilder slow_block(this);
    environment()->Push(jsgraph()->TheHoleConstant());
    slow_block.BeginBlock();
    environment()->Pop();
    fast_block.BeginBlock();

    // Perform checks whether the fast mode applies, by looking for any
    // extension object which might shadow the optimistic declaration.
    for (int depth = 0; bitset != 0; bitset >>= 1, depth++) {
      if ((bitset & 1) == 0) continue;
      Node* load = NewNode(
          javascript()->LoadContext(depth, Context::EXTENSION_INDEX, false),
          current_context());
      Node* check = NewNode(javascript()->StrictEqual(), load,
                            jsgraph()->TheHoleConstant());
      fast_block.BreakUnless(check, BranchHint::kTrue);
    }

    // Fast case, because variable is not shadowed. Perform context slot load.
    Variable* local = variable->local_if_not_shadowed();
    DCHECK(local->location() == VariableLocation::CONTEXT);  // Must be context.
    Node* fast = BuildVariableLoad(local, bailout_id, states, feedback, combine,
                                   typeof_mode);
    environment()->Push(fast);
    slow_block.Break();
    environment()->Pop();
    fast_block.EndBlock();

    // Slow case, because variable potentially shadowed. Perform dynamic lookup.
    const Operator* op = javascript()->LoadDynamic(name, typeof_mode);
    Node* slow = NewNode(op, BuildLoadFeedbackVector(), current_context());
    states.AddToNode(slow, bailout_id, combine);
    environment()->Push(slow);
    slow_block.EndBlock();

    return environment()->Pop();
  }

  return nullptr;
}
