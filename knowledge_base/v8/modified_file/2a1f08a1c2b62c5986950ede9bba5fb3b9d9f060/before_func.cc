void HGraph::AssignDominators() {
  HPhase phase("Assign dominators", this);
  for (int i = 0; i < blocks_.length(); ++i) {
    if (blocks_[i]->IsLoopHeader()) {
      // Only the first predecessor of a loop header is from outside the loop.
      // All others are back edges, and thus cannot dominate the loop header.
      blocks_[i]->AssignCommonDominator(blocks_[i]->predecessors()->first());
    } else {
      for (int j = 0; j < blocks_[i]->predecessors()->length(); ++j) {
        blocks_[i]->AssignCommonDominator(blocks_[i]->predecessors()->at(j));
      }
    }
  }
}
