void SILFunction::verifyCriticalEdges() const {
#ifdef NDEBUG
  if (!getModule().getOptions().VerifyAll)
    return;
#endif
  SILVerifier(*this, /*SingleFunction=*/true).verifyBranches(this);
}
