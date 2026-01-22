Reduction MachineOperatorReducer::ReduceWord32Comparisons(Node* node) {
  DCHECK(node->opcode() == IrOpcode::kInt32LessThan ||
         node->opcode() == IrOpcode::kInt32LessThanOrEqual ||
         node->opcode() == IrOpcode::kUint32LessThan ||
         node->opcode() == IrOpcode::kUint32LessThanOrEqual);
  Int32BinopMatcher m(node);
  // (x >> K) < (y >> K) => x < y   if only zeros shifted out
  if (m.left().op() == machine()->Word32SarShiftOutZeros() &&
      m.right().op() == machine()->Word32SarShiftOutZeros()) {
    Int32BinopMatcher mleft(m.left().node());
    Int32BinopMatcher mright(m.right().node());
    if (mleft.right().HasResolvedValue() &&
        mright.right().Is(mleft.right().ResolvedValue())) {
      node->ReplaceInput(0, mleft.left().node());
      node->ReplaceInput(1, mright.left().node());
      return Changed(node);
    }
  }
  // Simplifying (x >> n) <= k into x <= (k << n), with "k << n" being
  // computed here at compile time.
  if (m.right().HasResolvedValue() &&
      m.left().op() == machine()->Word32SarShiftOutZeros() &&
      m.left().node()->UseCount() == 1) {
    uint32_t right = m.right().ResolvedValue();
    Int32BinopMatcher mleft(m.left().node());
    if (mleft.right().HasResolvedValue()) {
      auto shift = mleft.right().ResolvedValue();
      if (CanRevertLeftShiftWithRightShift(right, shift)) {
        node->ReplaceInput(0, mleft.left().node());
        node->ReplaceInput(1, Int32Constant(right << shift));
        return Changed(node);
      }
    }
  }
  // Simplifying k <= (x >> n) into (k << n) <= x, with "k << n" being
  // computed here at compile time.
  if (m.left().HasResolvedValue() &&
      m.right().op() == machine()->Word32SarShiftOutZeros() &&
      m.right().node()->UseCount() == 1) {
    uint32_t left = m.left().ResolvedValue();
    Int32BinopMatcher mright(m.right().node());
    if (mright.right().HasResolvedValue()) {
      auto shift = mright.right().ResolvedValue();
      if (CanRevertLeftShiftWithRightShift(left, shift)) {
        node->ReplaceInput(0, Int32Constant(left << shift));
        node->ReplaceInput(1, mright.left().node());
        return Changed(node);
      }
    }
  }
  return NoChange();
}
