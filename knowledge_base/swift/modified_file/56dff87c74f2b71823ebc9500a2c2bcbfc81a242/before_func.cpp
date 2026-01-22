void SILGenFunction::checkForImportedUsedConformances(Type type) {
  // Recognize _BridgedNSError, which must pull in its witness table for
  // dynamic casts to work
  if (auto bridgedNSErrorProtocol =
          getASTContext().getProtocol(KnownProtocolKind::BridgedNSError)) {
    if (auto nominalDecl = type->getAnyNominal()) {
      auto conformances = nominalDecl->getAllConformances();
      for (auto conformance : conformances) {
        if (conformance->getProtocol() == bridgedNSErrorProtocol) {
          SGM.useConformance(ProtocolConformanceRef(conformance));
        }
      }
    }
  }
}
