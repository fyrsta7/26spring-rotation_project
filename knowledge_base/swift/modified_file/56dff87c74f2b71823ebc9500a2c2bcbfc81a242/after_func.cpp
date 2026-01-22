void SILGenFunction::checkForImportedUsedConformances(Type type) {
  // Recognize _BridgedNSError, which must pull in its witness table for
  // dynamic casts to work
  if (auto bridgedNSErrorProtocol =
          getASTContext().getProtocol(KnownProtocolKind::BridgedNSError)) {
    if (auto nominalDecl = type->getAnyNominal()) {
      SmallVector<ProtocolConformance *, 4> conformances;
      if (nominalDecl->lookupConformance(
              SGM.SwiftModule, bridgedNSErrorProtocol, conformances)) {
        SGM.useConformance(ProtocolConformanceRef(conformances.front()));
      }
    }
  }
}
