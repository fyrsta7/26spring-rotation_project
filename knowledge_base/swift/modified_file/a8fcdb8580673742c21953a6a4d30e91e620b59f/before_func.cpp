bool DisjunctionStep::shortCircuitDisjunctionAt(
    Constraint *currentChoice, Constraint *lastSuccessfulChoice) const {
  auto &ctx = CS.getASTContext();

  // Anything without a fix is better than anything with a fix.
  if (currentChoice->getFix() && !lastSuccessfulChoice->getFix())
    return true;

  if (ctx.TypeCheckerOpts.DisableConstraintSolverPerformanceHacks)
    return false;

  if (auto restriction = currentChoice->getRestriction()) {
    // Non-optional conversions are better than optional-to-optional
    // conversions.
    if (*restriction == ConversionRestrictionKind::OptionalToOptional)
      return true;

    // Array-to-pointer conversions are better than inout-to-pointer
    // conversions.
    if (auto successfulRestriction = lastSuccessfulChoice->getRestriction()) {
      if (*successfulRestriction == ConversionRestrictionKind::ArrayToPointer &&
          *restriction == ConversionRestrictionKind::InoutToPointer)
        return true;
    }
  }

  // Implicit conversions are better than checked casts.
  if (currentChoice->getKind() == ConstraintKind::CheckedCast)
    return true;

  // Extra checks for binding of operators
  if (currentChoice->getKind() == ConstraintKind::BindOverload &&
      currentChoice->getOverloadChoice().getDecl()->isOperator() &&
      lastSuccessfulChoice->getKind() == ConstraintKind::BindOverload &&
      lastSuccessfulChoice->getOverloadChoice().getDecl()->isOperator()) {

    // If we have a SIMD operator, and the prior choice was not a SIMD
    // Operator, we're done.
    if (isSIMDOperator(currentChoice->getOverloadChoice().getDecl()) &&
        !isSIMDOperator(lastSuccessfulChoice->getOverloadChoice().getDecl()))
      return true;
  }
  return false;
}
