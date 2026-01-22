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
  Function *DTorFn =dyn_cast<Function>(CS->getOperand(DTorSlotOfHeapMeatadata));
  if (DTorFn == 0 || DTorFn->mayBeOverridden() || DTorFn->hasExternalLinkage())
    return DtorKind::Unknown;
  
  // Okay, we have a body, and we can trust it.  The first argument
  assert(DTorFn->arg_size() == 1 && !DTorFn->isVarArg() &&
         "expected a single object argument to destructors");

  // Scan the body of the function, looking for anything scary.
  for (BasicBlock &BB : *DTorFn) {
    for (Instruction &I : BB) {
      // Ignore all instructions with side effects.
      if (!I.mayHaveSideEffects()) continue;
      
      // TODO: Ignore instructions that just side effect the object
      // (stores/loads/memcpy/etc).
        
        
      // Okay, the function has some side effects, if it doesn't capture the
      // object argument, at least that is something.
      return DTorFn->doesNotCapture(0) ? DtorKind::NoEscape : DtorKind::Unknown;
    }
  }
  
  // If we didn't find any side effects, we win.
  return DtorKind::NoSideEffects;
}
