void SILPerformanceInliner::inlineDevirtualizeAndSpecialize(
                                                          SILFunction *Caller,
                                                         SILModuleTransform *MT,
                                                         CallGraphAnalysis *CGA,
                                                          DominanceAnalysis *DA,
                                                          SILLoopAnalysis *LA) {
  assert(Caller->isDefinition() &&
         "Expected only defined functions in the call graph!");

  llvm::SmallVector<SILFunction *, 4> WorkList;
  WorkList.push_back(Caller);

  auto &CG = CGA->getOrBuildCallGraph();
  OriginMap.clear();
  RemovedApplies.clear();

  while (!WorkList.empty()) {
    llvm::SmallVector<FullApplySite, 4> WorkItemApplies;
    collectAllAppliesInFunction(WorkList.back(), WorkItemApplies);

    // Devirtualize and specialize any applies we've collected,
    // and collect new functions we should inline into as we do
    // so.
    llvm::SmallVector<SILFunction *, 4> NewFuncs;
    if (devirtualizeAndSpecializeApplies(WorkItemApplies, CGA, MT, NewFuncs)) {
      WorkList.insert(WorkList.end(), NewFuncs.begin(), NewFuncs.end());
      NewFuncs.clear();
    }
    assert(WorkItemApplies.empty() && "Expected all applies to be processed!");

    // We want to inline into each function on the worklist, starting
    // with any new ones that were exposed as a result of
    // devirtualization (to insure we're inlining into callees first).
    //
    // After inlining, we may have new opportunities for
    // devirtualization, e.g. as a result of exposing the dynamic type
    // of an object. When those opportunities arise we want to attempt
    // devirtualization and then again attempt to inline into the
    // newly exposed functions, etc. until we're back to the function
    // we began with.
    auto *Initial = WorkList.back();

    // In practice we rarely exceed 5, but in a perf test we iterate 51 times.
    const unsigned MaxLaps = 150;
    unsigned Lap = 0;
    while (1) {
      auto *WorkItem = WorkList.back();
      assert(WorkItem->isDefinition() &&
        "Expected function definition on work list!");

      // Devirtualization and specialization might have exposed new
      // function references. We want to inline within those functions
      // before inlining within our original function.
      //
      // Inlining in turn might result in new applies that we should
      // consider for devirtualization and specialization.
      llvm::SmallVector<FullApplySite, 4> NewApplies;
      if (inlineCallsIntoFunction(WorkItem, DA, LA, CG, NewApplies)) {
        // Invalidate analyses, but lock the call graph since we
        // maintain it.
        CGA->lockInvalidation();
        MT->invalidateAnalysis(WorkItem, SILAnalysis::PreserveKind::Nothing);
        CGA->unlockInvalidation();

        // FIXME: Update inlineCallsIntoFunction to collect all
        //        remaining applies after inlining, not just those
        //        resulting from inlining code.
        llvm::SmallVector<FullApplySite, 4> WorkItemApplies;
        collectAllAppliesInFunction(WorkItem, WorkItemApplies);

        if (devirtualizeAndSpecializeApplies(WorkItemApplies, CGA, MT,
                                             NewFuncs)) {
          WorkList.insert(WorkList.end(), NewFuncs.begin(), NewFuncs.end());
          NewFuncs.clear();
          assert(WorkItemApplies.empty() &&
                 "Expected all applies to be processed!");
        } else if (WorkItem == Initial) {
          break;
        } else {
          WorkList.pop_back();
        }
      } else if (WorkItem == Initial) {
        break;
      } else {
        WorkList.pop_back();
      }

      Lap++;
      if (Lap > MaxLaps)
      // It's possible to construct real code where this will hit, but
      // it's more likely that there is an issue tracking recursive
      // inlining, in which case we want to know about it in internal
      // builds, and not hang on bots or user machines.
      assert(Lap <= MaxLaps && "Possible bug tracking recursion!");
      // Give up and move along.
      if (Lap > MaxLaps) {
        while (WorkList.back() != Initial)
          WorkList.pop_back();
        break;
      }
    }

    assert(WorkList.back() == Initial &&
           "Expected to exit with same element on top of stack!" );
    WorkList.pop_back();
  }
}
