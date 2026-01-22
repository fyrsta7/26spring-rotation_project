  void AnalyzeDominatedBlocks(HBasicBlock* root, State* initial) {
    InitializeStates();
    SetStateAt(root, initial);

    // Iterate all dominated blocks starting from the given start block.
    for (int i = root->block_id(); i < graph_->blocks()->length(); i++) {
      HBasicBlock* block = graph_->blocks()->at(i);

      // Skip blocks not dominated by the root node.
      if (SkipNonDominatedBlock(root, block)) continue;
      State* state = StateAt(block);

      if (block->IsLoopHeader()) {
        // Apply loop effects before analyzing loop body.
        ComputeLoopEffects(block)->Apply(state);
      } else {
        // Must have visited all predecessors before this block.
        CheckPredecessorCount(block);
      }

      // Go through all instructions of the current block, updating the state.
      for (HInstructionIterator it(block); !it.Done(); it.Advance()) {
        state = state->Process(it.Current(), zone_);
      }

      // Propagate the block state forward to all successor blocks.
      int max = block->end()->SuccessorCount();
      for (int i = 0; i < max; i++) {
        HBasicBlock* succ = block->end()->SuccessorAt(i);
        IncrementPredecessorCount(succ);
        if (StateAt(succ) == NULL) {
          // This is the first state to reach the successor.
          if (max == 1 && succ->predecessors()->length() == 1) {
            // Optimization: successor can inherit this state.
            SetStateAt(succ, state);
          } else {
            // Successor needs a copy of the state.
            SetStateAt(succ, state->Copy(succ, zone_));
          }
        } else {
          // Merge the current state with the state already at the successor.
          SetStateAt(succ, state->Merge(succ, StateAt(succ), zone_));
        }
      }
    }
  }
