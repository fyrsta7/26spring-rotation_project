void swift::rewriting::applyInverses(
    ASTContext &ctx,
    ArrayRef<Type> gps,
    ArrayRef<InverseRequirement> inverseList,
    SmallVectorImpl<StructuralRequirement> &result,
    SmallVectorImpl<RequirementError> &errors) {

  if (!ctx.LangOpts.hasFeature(Feature::NoncopyableGenerics))
    return;

  // Summarize the inverses and flag ones that are incorrect.
  llvm::DenseMap<CanType, InvertibleProtocolSet> inverses;
  for (auto inverse : inverseList) {
    auto canSubject = inverse.subject->getCanonicalType();

    // WARNING: possible quadratic behavior, but should be OK in practice.
    auto notInScope = llvm::none_of(gps, [=](Type t) {
      return t->getCanonicalType() == canSubject;
    });

    // If the inverse is on a subject that wasn't permitted by our caller, then
    // remove and diagnose as an error. This can happen when an inner context
    // has a constraint on some outer generic parameter, e.g.,
    //
    //     protocol P {
    //       func f() where Self: ~Copyable
    //     }
    //
    if (notInScope) {
      errors.push_back(
          RequirementError::forInvalidInverseOuterSubject(inverse));
      continue;
    }

    auto state = inverses.getOrInsertDefault(canSubject);

    // Check if this inverse has already been seen.
    auto inverseKind = inverse.getKind();
    if (state.contains(inverseKind)) {
      errors.push_back(
          RequirementError::forRedundantInverseRequirement(inverse));
      continue;
    }

    state.insert(inverseKind);
    inverses[canSubject] = state;
  }

  // Fast-path: if there are no valid inverses, then there are no requirements
  // to be removed.
  if (inverses.empty())
    return;

  // Scan the structural requirements and cancel out any inferred requirements
  // based on the inverses we saw.
  result.erase(llvm::remove_if(result, [&](StructuralRequirement structReq) {
    auto req = structReq.req;

    if (req.getKind() != RequirementKind::Conformance)
      return false;

    // Only consider requirements from defaults-expansion...
    if (!structReq.fromDefault)
      return false;

    // involving an invertible protocol.
    llvm::Optional<InvertibleProtocolKind> proto;
    if (auto kp = req.getProtocolDecl()->getKnownProtocolKind())
      if (auto ip = getInvertibleProtocolKind(*kp))
        proto = *ip;

    if (!proto) {
      assert(false && "suspicious!");
      return false;
    }

    // See if this subject is in-scope.
    auto subject = req.getFirstType()->getCanonicalType();
    auto result = inverses.find(subject);
    if (result == inverses.end())
      return false;

    // We now have found the inferred constraint 'Subject : Proto'.
    // So, remove it if we have recorded a 'Subject : ~Proto'.
    auto recordedInverses = result->getSecond();
    return recordedInverses.contains(*proto);
  }), result.end());
}
