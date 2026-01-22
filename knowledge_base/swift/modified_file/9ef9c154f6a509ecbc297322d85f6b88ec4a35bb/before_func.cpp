void MandatoryGenericSpecializer::run() {
  SILModule *module = getModule();

  if (!module->getOptions().EnablePerformanceAnnotations)
    return;

  ClassHierarchyAnalysis *cha = getAnalysis<ClassHierarchyAnalysis>();
  
  llvm::SmallVector<SILFunction *, 8> workList;
  llvm::SmallPtrSet<SILFunction *, 16> visited;
  
  // Look for performance-annotated functions.
  for (SILFunction &function : *module) {
    if (function.getPerfConstraints() != PerformanceConstraints::None) {
      workList.push_back(&function);
      visited.insert(&function);
    }
  }
  
  while (!workList.empty()) {
    SILFunction *func = workList.pop_back_val();
    module->linkFunction(func, SILModule::LinkingMode::LinkAll);
    if (!func->isDefinition())
      continue;

    // Perform generic specialization and other related optimzations.
    bool changed = optimize(func, cha);

    if (changed)
      invalidateAnalysis(func, SILAnalysis::InvalidationKind::Everything);

    // Continue specializing called functions.
    for (SILBasicBlock &block : *func) {
      for (SILInstruction &inst : block) {
        if (auto as = ApplySite::isa(&inst)) {
          if (SILFunction *callee = as.getReferencedFunctionOrNull()) {
            if (visited.insert(callee).second)
              workList.push_back(callee);
          }
        }
      }
    }
  }
}
