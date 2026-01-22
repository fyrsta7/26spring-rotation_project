static bool checkAllocBoxUses(AllocBoxInst *ABI, ValueBase *V,
                              SmallVectorImpl<SILInstruction*> &Users,
                              SmallVectorImpl<SILInstruction*> &Releases) {
  for (auto UI : V->getUses()) {
    auto *User = UI->getUser();
    
    // These instructions do not cause the box's address to escape.
    if (isa<StrongRetainInst>(User) ||
        isa<CopyAddrInst>(User) ||
        isa<LoadInst>(User) ||
        isa<InitializeVarInst>(User) ||
        isa<ProtocolMethodInst>(User) ||
        (isa<StoreInst>(User) && UI->getOperandNumber() == 1) ||
        (isa<AssignInst>(User) && UI->getOperandNumber() == 1)) {
      Users.push_back(User);
      continue;
    }
    
    // Release doesn't either, but we want to keep track of where this value
    // gets released.
    if (isa<StrongReleaseInst>(User) || isa<DeallocBoxInst>(User)) {
      Releases.push_back(User);
      Users.push_back(User);
      continue;
    }

    // These instructions only cause the alloc_box to escape if they are used in
    // a way that escapes.  Recursively check that the uses of the instruction
    // don't escape and collect all of the uses of the value.
    if (isa<StructElementAddrInst>(User) || isa<TupleElementAddrInst>(User) ||
        isa<ProjectExistentialInst>(User)) {
      Users.push_back(User);
      if (checkAllocBoxUses(ABI, User, Users, Releases))
        return true;
      continue;
    }
    
    // apply and partial_apply instructions do not capture the pointer when
    // it is passed through [inout] arguments or for indirect returns.
    if (auto apply = dyn_cast<ApplyInst>(User)) {
      if (apply->getFunctionTypeInfo()
            ->getParameters()[UI->getOperandNumber()-1].isIndirect())
        continue;
    }
    if (auto partialApply = dyn_cast<PartialApplyInst>(User)) {
      if (partialApply->getFunctionTypeInfo()
            ->getParameters()[UI->getOperandNumber()-1].isIndirect())
        continue;
      
    }

    // Otherwise, this looks like it escapes.
    DEBUG(llvm::errs() << "*** Failed to promote alloc_box: " << *ABI
          << "    Due to user: " << *User << "\n");
    
    return true;
  }
  
  return false;
}
