static Optional<SolutionStep> getNextSolutionStep(ConstraintSystem &cs) {
  // If there are any potential bindings to explore, do it now.
  if (cs.hasPotentialBindings()) {
    return SolutionStep(SolutionStepKind::ExploreBindings);
  }
  
  SmallVector<TypeVariableConstraints, 16> typeVarConstraints;
  cs.collectConstraintsForTypeVariables(typeVarConstraints);

  // If there are any type variables that we can definitively solve,
  // do so now.
  if (bindDefinitiveTypeVariables(cs, typeVarConstraints)) {
    return SolutionStep(SolutionStepKind::Simplify);
  }

  // If there are any unresolved overload sets, resolve one now.
  // FIXME: This is terrible for performance.
  if (cs.getNumUnresolvedOverloadSets() > 0) {
    // Resolve the first unresolved overload set.
    return SolutionStep(0);
  }

  // Try to determine a binding for a type variable.
  if (auto binding = resolveTypeVariable(cs, typeVarConstraints,
                                         /*onlyDefinitive=*/false)) {
    using std::get;
    return SolutionStep(get<0>(*binding), get<1>(*binding), get<2>(*binding));
  }

  // We're out of ideas.
  return Nothing;
}
