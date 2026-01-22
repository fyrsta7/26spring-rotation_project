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
    // Find the overload set with the minimum number of overloads.
    unsigned minSize = cs.getUnresolvedOverloadSet(0)->getChoices().size();
    unsigned minIdx = 0;
    if (minSize > 2) {
      for (unsigned i = 1, n = cs.getNumUnresolvedOverloadSets(); i < n; ++i) {
        unsigned newSize = cs.getUnresolvedOverloadSet(i)->getChoices().size();
        if (newSize < minSize) {
          minSize = newSize;
          minIdx = i;

          if (minSize == 2)
            break;
        }
      }
    }

    // Resolve the unresolved overload set with the minimum number of overloads.
    return SolutionStep(minIdx);
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
