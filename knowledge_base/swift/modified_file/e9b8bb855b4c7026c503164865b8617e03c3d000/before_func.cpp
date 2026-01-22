bool LSLocation::isNonEscapingLocalLSLocation(SILFunction *Fn,
                                              EscapeAnalysis *EA) {
  auto *ConGraph = EA->getConnectionGraph(Fn);
  // An alloc_stack is definitely dead at the end of the function.
  if (isa<AllocStackInst>(Base))
    return true;
  // For other allocations we ask escape analysis.
  if (isa<AllocationInst>(Base)) {
    auto *Node = ConGraph->getNodeOrNull(Base, EA);
    if (Node && !Node->escapes()) {
      return true;
    }
  }
  return false;
}
