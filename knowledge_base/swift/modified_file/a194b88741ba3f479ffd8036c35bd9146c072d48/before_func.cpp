static bool CCPFunctionBody(SILFunction &F, SILModule &M) {
  DEBUG(llvm::errs() << "*** ConstPropagation processing: " << F.getName()
        << "\n");

  // Initialize the worklist to all of the instructions ready to process...
  std::set<SILInstruction*> WorkList;
  for (auto &BB : F) {
    for(auto &I : BB) {
      WorkList.insert(&I);
    }
  }

  // Try to fold instructions in the list one by one.
  bool Folded = false;
  while (!WorkList.empty()) {
    SILInstruction *I = *WorkList.begin();
    WorkList.erase(WorkList.begin());

    if (!I->use_empty())

      // Try to fold the instruction.
      if (SILInstruction *C = constantFoldInstruction(*I, M)) {
        // The users could be constant propagatable now.
        for (auto UseI = I->use_begin(),
                  UseE = I->use_end(); UseI != UseE; ++UseI) {
          SILInstruction *User = cast<SILInstruction>(UseI.getUser());
          WorkList.insert(User);

          // Some constant users may indirectly cause folding of their users.
          if (isa<StructInst>(User) || isa<TupleInst>(User)) {
            for (auto UseUseI = User->use_begin(),
                 UseUseE = User->use_end(); UseUseI != UseUseE; ++UseUseI) {
              WorkList.insert(cast<SILInstruction>(UseUseI.getUser()));

            }
          }
        }

        // We were able to fold, so all users should use the new folded value.
        assert(I->getTypes().size() == 1 &&
               "Currently, we only support single result instructions.");
        SILValue(I).replaceAllUsesWith(C);

        // Remove the unused instruction.
        WorkList.erase(I);

        // Eagerly DCE.
        recursivelyDeleteTriviallyDeadInstructions(I);

        Folded = true;
        ++NumInstFolded;
      }
  }

  return false;
}
