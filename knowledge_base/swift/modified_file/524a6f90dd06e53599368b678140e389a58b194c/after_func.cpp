void SideEffectAnalysis::getEffects(FunctionEffects &FE, FullApplySite FAS) {
  CallGraph *CG = CGA->getCallGraphOrNull();
  if (CG) {
    getEffectsOfApply(FE, FAS, *CG, false);
    return;
  }
  FE.setWorstEffects();
}
