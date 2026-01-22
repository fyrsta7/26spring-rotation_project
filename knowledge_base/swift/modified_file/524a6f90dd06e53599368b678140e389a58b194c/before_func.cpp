void SideEffectAnalysis::getEffects(FunctionEffects &FE, FullApplySite FAS) {
  CallGraph &CG = CGA->getOrBuildCallGraph();
  getEffectsOfApply(FE, FAS, CG, false);
}
