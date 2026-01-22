static Optional<bool> shouldInlineGeneric(FullApplySite AI) {
  assert(!AI.getSubstitutions().empty() &&
         "Expected a generic apply");

  if (!EnableSILInliningOfGenerics)
    return false;

  // If all substitutions are concrete, then there is no need to perform the
  // generic inlining. Let the generic specializer create a specialized
  // function and then decide if it is beneficial to inline it.
  if (!hasArchetypes(AI.getSubstitutions()))
    return false;

  SILFunction *Callee = AI.getReferencedFunction();

  // Do not inline @_semantics functions when compiling the stdlib,
  // because they need to be preserved, so that the optimizer
  // can properly optimize a user code later.
  auto ModuleName = Callee->getModule().getSwiftModule()->getName().str();
  if (Callee->hasSemanticsAttrThatStartsWith("array.") &&
      (ModuleName == STDLIB_NAME || ModuleName == SWIFT_ONONE_SUPPORT))
    return false;

  // Do not inline into thunks.
  if (AI.getFunction()->isThunk())
    return false;

  // Always inline generic functions which are marked as
  // AlwaysInline or transparent.
  if (Callee->getInlineStrategy() == AlwaysInline || Callee->isTransparent())
    return true;

  // It is not clear yet if this function should be decided or not.
  return None;
}
