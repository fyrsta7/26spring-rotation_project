void MaglevGraphBuilder::BuildBranchIfRootConstant(ValueNode* node,
                                                   JumpType jump_type,
                                                   RootIndex root_index) {
  int fallthrough_offset = next_offset();
  int jump_offset = iterator_.GetJumpTargetOffset();
  BasicBlockRef* true_target = jump_type == kJumpIfTrue
                                   ? &jump_targets_[jump_offset]
                                   : &jump_targets_[fallthrough_offset];
  BasicBlockRef* false_target = jump_type == kJumpIfFalse
                                    ? &jump_targets_[jump_offset]
                                    : &jump_targets_[fallthrough_offset];
  BasicBlock* block = FinishBlock<BranchIfRootConstant>(
      {node}, true_target, false_target, root_index);
  if (jump_type == kJumpIfTrue) {
    block->control_node()
        ->Cast<BranchControlNode>()
        ->set_true_interrupt_correction(
            iterator_.GetRelativeJumpTargetOffset());
  } else {
    block->control_node()
        ->Cast<BranchControlNode>()
        ->set_false_interrupt_correction(
            iterator_.GetRelativeJumpTargetOffset());
  }
  MergeIntoFrameState(block, jump_offset);
  StartFallthroughBlock(fallthrough_offset, block);
}
