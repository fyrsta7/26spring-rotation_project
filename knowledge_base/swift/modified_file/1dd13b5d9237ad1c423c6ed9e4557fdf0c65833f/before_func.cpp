bool DSEContext::run() {
  // Walk over the function and find all the locations accessed by
  // this function.
  LSLocation::enumerateLSLocations(*F, LocationVault, LocToBitIndex, TE);

  // For all basic blocks in the function, initialize a BB state.
  //
  // DenseMap has a minimum size of 64, while many functions do not have more
  // than 64 basic blocks. Therefore, allocate the BlockState in a vector and
  // use pointer in BBToLocState to access them.
  for (auto &B : *F) {
    BlockStates.push_back(BlockState(&B));
    // Since we know all the locations accessed in this function, we can resize
    // the bit vector to the appropriate size.
    BlockStates.back().init(*this);
  }

  // Initialize the BBToLocState mapping.
  for (auto &S : BlockStates) {
    BBToLocState[S.getBB()] = &S;
  }

  // We perform dead store elimination in the following phases.
  //
  // Phase 1. we compute the max store set at the beginning of the basic block.
  //
  // Phase 2. we compute the genset and killset for every basic block.
  //
  // Phase 3. we run the data flow with the genset and killset until
  // BBWriteSetIns stop changing.
  //
  // Phase 4. we run the data flow for the last iteration and perform the DSE.
  //
  // Phase 5. we remove the dead stores.

  // Compute the max store set at the beginning of the basic block.
  //
  // This helps generating the genset and killset. If there is no way a
  // location can have an upward visible store at a particular point in the
  // basic block, we do not need to turn on the genset and killset for the
  // location.
  //
  // Turning on the genset and killset can be costly as it involves querying
  // the AA interface.
  auto *PO = PM->getAnalysis<PostOrderAnalysis>()->get(F);
  for (SILBasicBlock *B : PO->getPostOrder()) {
    processBasicBlockForMaxStoreSet(B);
  }

  // Generate the genset and killset for each basic block. We can process the
  // basic blocks in any order.
  for (auto &B : *F) {
    processBasicBlockForGenKillSet(&B);
  }

  // Process each basic block with the gen and kill set. Every time the
  // BBWriteSetIn of a basic block changes, the optimization is rerun on its
  // predecessors.
  llvm::SmallVector<SILBasicBlock *, 16> WorkList;
  for (SILBasicBlock *B : PO->getPostOrder()) {
    WorkList.push_back(B);
  }
  while (!WorkList.empty()) {
    SILBasicBlock *BB = WorkList.pop_back_val();
    if (processBasicBlockWithGenKillSet(BB)) {
      for (auto X : BB->getPreds())
        WorkList.push_back(X);
    }
  }

  // The data flow has stabilized, run one last iteration over all the basic
  // blocks and try to remove dead stores.
  for (SILBasicBlock *B : PO->getPostOrder()) {
    processBasicBlockForDSE(B);
  }

  // Finally, delete the dead stores and create the live stores.
  bool Changed = false;
  for (SILBasicBlock &BB : *F) {
    // Create the stores that are alive due to partial dead stores.
    for (auto &I : getBlockState(&BB)->LiveStores) {
      Changed = true;
      SILInstruction *IT = cast<SILInstruction>(I.first)->getNextNode();
      SILBuilderWithScope Builder(IT);
      Builder.createStore(I.first.getLoc().getValue(), I.second, I.first);
    }
    // Delete the dead stores.
    for (auto &I : getBlockState(&BB)->DeadStores) {
      Changed = true;
      DEBUG(llvm::dbgs() << "*** Removing: " << *I << " ***\n");
      // This way, we get rid of pass dependence on DCE.
      recursivelyDeleteTriviallyDeadInstructions(I, true);
    }
  }
  return Changed;
}
