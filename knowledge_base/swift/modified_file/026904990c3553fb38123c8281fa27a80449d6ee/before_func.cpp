ResolvedMemberResult
swift::resolveValueMember(DeclContext &DC, Type BaseTy, DeclName Name) {
  ResolvedMemberResult Result;
  ConstraintSystem CS(&DC, None);

  // Look up all members of BaseTy with the given Name.
  MemberLookupResult LookupResult = CS.performMemberLookup(
      ConstraintKind::ValueMember, DeclNameRef(Name), BaseTy,
      FunctionRefKind::SingleApply, CS.getConstraintLocator({}), false);

  // Keep track of all the unviable members.
  for (auto Can : LookupResult.UnviableCandidates)
    Result.Impl->AllDecls.push_back(Can.getDecl());

  // Keep track of the start of viable choices.
  Result.Impl->ViableStartIdx = Result.Impl->AllDecls.size();

  // If no viable members, we are done.
  if (LookupResult.ViableCandidates.empty())
    return Result;

  // Try to figure out the best overload.
  ConstraintLocator *Locator = CS.getConstraintLocator({});
  TypeVariableType *TV = CS.createTypeVariable(Locator,
                                               TVO_CanBindToLValue |
                                               TVO_CanBindToNoEscape);
  CS.addOverloadSet(TV, LookupResult.ViableCandidates, &DC, Locator);
  Optional<Solution> OpSolution = CS.solveSingle();
  ValueDecl *Selected = nullptr;
  if (OpSolution.hasValue()) {
    Selected = OpSolution.getValue().overloadChoices[Locator].choice.getDecl();
  }
  for (OverloadChoice& Choice : LookupResult.ViableCandidates) {
    ValueDecl *VD = Choice.getDecl();

    // If this VD is the best overload, keep track of its index.
    if (VD == Selected)
      Result.Impl->BestIdx = Result.Impl->AllDecls.size();
    Result.Impl->AllDecls.push_back(VD);
  }
  return Result;
}
