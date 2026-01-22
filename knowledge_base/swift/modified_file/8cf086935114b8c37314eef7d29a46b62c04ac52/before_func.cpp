static SILFunction *getCalleeFunction(
    SILFunction *F, FullApplySite AI, bool &IsThick,
    SmallVectorImpl<std::pair<SILValue, ParameterConvention>> &CaptureArgs,
    SmallVectorImpl<SILValue> &FullArgs, PartialApplyInst *&PartialApply,
    SILModule::LinkingMode Mode) {
  IsThick = false;
  PartialApply = nullptr;
  CaptureArgs.clear();
  FullArgs.clear();

  for (const auto &Arg : AI.getArguments())
    FullArgs.push_back(Arg);
  SILValue CalleeValue = AI.getCallee();

  if (auto *LI = dyn_cast<LoadInst>(CalleeValue)) {
    // Conservatively only see through alloc_box; we assume this pass is run
    // immediately after SILGen
    auto *PBI = dyn_cast<ProjectBoxInst>(LI->getOperand());
    if (!PBI)
      return nullptr;
    auto *ABI = dyn_cast<AllocBoxInst>(PBI->getOperand());
    if (!ABI)
      return nullptr;
    // Ensure there are no other uses of alloc_box than the project_box and
    // retains, releases.
    for (Operand *ABIUse : ABI->getUses())
      if (ABIUse->getUser() != PBI &&
          !isa<StrongRetainInst>(ABIUse->getUser()) &&
          !isa<StrongReleaseInst>(ABIUse->getUser()))
        return nullptr;

    // Scan forward from the alloc box to find the first store, which
    // (conservatively) must be in the same basic block as the alloc box
    StoreInst *SI = nullptr;
    for (auto I = SILBasicBlock::iterator(ABI), E = I->getParent()->end();
         I != E; ++I) {
      // If we find the load instruction first, then the load is loading from
      // a non-initialized alloc; this shouldn't really happen but I'm not
      // making any assumptions
      if (&*I == LI)
        return nullptr;
      if ((SI = dyn_cast<StoreInst>(I)) && SI->getDest() == PBI) {
        // We found a store that we know dominates the load; now ensure there
        // are no other uses of the project_box except loads.
        for (Operand *PBIUse : PBI->getUses())
          if (PBIUse->getUser() != SI && !isa<LoadInst>(PBIUse->getUser()))
            return nullptr;
        // We can conservatively see through the store
        break;
      }
    }
    if (!SI)
      return nullptr;
    CalleeValue = SI->getSrc();
  }

  // PartialApply/ThinToThick -> ConvertFunction patterns are generated
  // by @noescape closures.
  //
  // FIXME: We don't currently handle mismatched return types, however, this
  // would be a good optimization to handle and would be as simple as inserting
  // a cast.
  auto skipFuncConvert = [](SILValue CalleeValue) {
    // We can also allow a thin @escape to noescape conversion as such:
    // %1 = function_ref @thin_closure_impl : $@convention(thin) () -> ()
    // %2 = convert_function %1 :
    //      $@convention(thin) () -> () to $@convention(thin) @noescape () -> ()
    // %3 = thin_to_thick_function %2 :
    //  $@convention(thin) @noescape () -> () to
    //            $@noescape @callee_guaranteed () -> ()
    // %4 = apply %3() : $@noescape @callee_guaranteed () -> ()
    if (auto *ThinToNoescapeCast = dyn_cast<ConvertFunctionInst>(CalleeValue)) {
      auto FromCalleeTy =
          ThinToNoescapeCast->getOperand()->getType().castTo<SILFunctionType>();
      if (FromCalleeTy->getExtInfo().hasContext())
        return CalleeValue;
      auto ToCalleeTy = ThinToNoescapeCast->getType().castTo<SILFunctionType>();
      auto EscapingCalleeTy = ToCalleeTy->getWithExtInfo(
          ToCalleeTy->getExtInfo().withNoEscape(false));
      if (FromCalleeTy != EscapingCalleeTy)
        return CalleeValue;
      return ThinToNoescapeCast->getOperand();
    }

    auto *CFI = dyn_cast<ConvertEscapeToNoEscapeInst>(CalleeValue);
    if (!CFI)
      return CalleeValue;

    // TODO: Handle argument conversion. All the code in this file needs to be
    // cleaned up and generalized. The argument conversion handling in
    // optimizeApplyOfConvertFunctionInst should apply to any combine
    // involving an apply, not just a specific pattern.
    //
    // For now, just handle conversion that doesn't affect argument types,
    // return types, or throws. We could trivially handle any other
    // representation change, but the only one that doesn't affect the ABI and
    // matters here is @noescape, so just check for that.
    auto FromCalleeTy = CFI->getOperand()->getType().castTo<SILFunctionType>();
    auto ToCalleeTy = CFI->getType().castTo<SILFunctionType>();
    auto EscapingCalleeTy =
      ToCalleeTy->getWithExtInfo(ToCalleeTy->getExtInfo().withNoEscape(false));
    if (FromCalleeTy != EscapingCalleeTy)
      return CalleeValue;

    return CFI->getOperand();
  };

  // Look through a escape to @noescape conversion.
  CalleeValue = skipFuncConvert(CalleeValue);

  // We are allowed to see through exactly one "partial apply" instruction or
  // one "thin to thick function" instructions, since those are the patterns
  // generated when using auto closures.
  if (auto *PAI = dyn_cast<PartialApplyInst>(CalleeValue)) {

    // Collect the applied arguments and their convention.
    collectPartiallyAppliedArguments(PAI, CaptureArgs, FullArgs);

    CalleeValue = PAI->getCallee();
    IsThick = true;
    PartialApply = PAI;
  } else if (auto *TTTFI = dyn_cast<ThinToThickFunctionInst>(CalleeValue)) {
    CalleeValue = TTTFI->getOperand();
    IsThick = true;
  }

  CalleeValue = skipFuncConvert(CalleeValue);

  auto *FRI = dyn_cast<FunctionRefInst>(CalleeValue);
  if (!FRI)
    return nullptr;

  SILFunction *CalleeFunction = FRI->getReferencedFunction();

  switch (CalleeFunction->getRepresentation()) {
  case SILFunctionTypeRepresentation::Thick:
  case SILFunctionTypeRepresentation::Thin:
  case SILFunctionTypeRepresentation::Method:
  case SILFunctionTypeRepresentation::Closure:
  case SILFunctionTypeRepresentation::WitnessMethod:
    break;
    
  case SILFunctionTypeRepresentation::CFunctionPointer:
  case SILFunctionTypeRepresentation::ObjCMethod:
  case SILFunctionTypeRepresentation::Block:
    return nullptr;
  }

  // If CalleeFunction is a declaration, see if we can load it. If we fail to
  // load it, bail.
  if (CalleeFunction->empty()
      && !AI.getModule().linkFunction(CalleeFunction, Mode))
    return nullptr;

  // If the CalleeFunction is a not-transparent definition, we can not process
  // it.
  if (CalleeFunction->isTransparent() == IsNotTransparent)
    return nullptr;

  if (F->isSerialized() &&
      !CalleeFunction->hasValidLinkageForFragileInline()) {
    if (!CalleeFunction->hasValidLinkageForFragileRef()) {
      llvm::errs() << "caller: " << F->getName() << "\n";
      llvm::errs() << "callee: " << CalleeFunction->getName() << "\n";
      llvm_unreachable("Should never be inlining a resilient function into "
                       "a fragile function");
    }
    return nullptr;
  }

  return CalleeFunction;
}
