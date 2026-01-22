void SILFunction::verifyCriticalEdges() const {
  SILVerifier(*this, /*SingleFunction=*/true).verifyBranches(this);
}
