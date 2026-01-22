static DtorKind analyzeDestructor(Value *P) {
  // If we have a null pointer for the metadata info, the dtor has no side
  // effects.  Actually, the final release would crash.  This is really only
  // useful for writing testcases.
  if (isa<ConstantPointerNull>(P->stripPointerCasts()))
    return DtorKind::NoSideEffects;
    
  // We have to have a known heap metadata value, reject dynamically computed
  // ones, or places 
  GlobalVariable *GV = dyn_cast<GlobalVariable>(P->stripPointerCasts());
  if (GV == 0 || GV->mayBeOverridden()) return DtorKind::Unknown;
  
  ConstantStruct *CS = dyn_cast_or_null<ConstantStruct>(GV->getInitializer());
  if (CS == 0 || CS->getNumOperands() == 0) return DtorKind::Unknown;
  
  // FIXME: Would like to abstract the dtor slot (#0) out from this to somewhere
  // unified.
  enum { DTorSlotOfHeapMeatadata = 0 };
  Function *DtorFn =dyn_cast<Function>(CS->getOperand(DTorSlotOfHeapMeatadata));
  if (DtorFn == 0 || DtorFn->mayBeOverridden() || DtorFn->hasExternalLinkage())
    return DtorKind::Unknown;
  
  // Okay, we have a body, and we can trust it.  If the function is marked
  // readonly, then we know it can't have any interesting side effects, so we
  // don't need to analyze it at all.
  if (DtorFn->onlyReadsMemory())
    return DtorKind::NoSideEffects;
  
  // The first argument is the object being destroyed.
  assert(DtorFn->arg_size() == 1 && !DtorFn->isVarArg() &&
         "expected a single object argument to destructors");

  // Scan the body of the function, looking for anything scary.
  for (BasicBlock &BB : *DtorFn) {
    for (Instruction &I : BB) {
      // Ignore all instructions with side effects.
      if (!I.mayHaveSideEffects()) continue;
      
      // TODO: Ignore instructions that just side effect the object
      // (stores/loads/memcpy/etc).
        
        
      // Okay, the function has some side effects, if it doesn't capture the
      // object argument, at least that is something.
      return DtorFn->doesNotCapture(0) ? DtorKind::NoEscape : DtorKind::Unknown;
    }
  }
  
  // If we didn't find any side effects, we win.
  return DtorKind::NoSideEffects;
}
