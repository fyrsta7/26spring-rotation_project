void swift::performLLVMOptimizations(IRGenOptions &Opts, llvm::Module *Module,
                                     llvm::TargetMachine *TargetMachine) {
  // Set up a pipeline.
  PassManagerBuilder PMBuilder;

  if (Opts.Optimize && !Opts.DisableLLVMOptzns) {
    PMBuilder.OptLevel = 3;
    PMBuilder.Inliner = llvm::createFunctionInliningPass(200);
    PMBuilder.SLPVectorize = true;
    PMBuilder.LoopVectorize = true;
  } else {
    PMBuilder.OptLevel = 0;
    if (!Opts.DisableLLVMOptzns)
      PMBuilder.Inliner =
        llvm::createAlwaysInlinerPass(/*insertlifetime*/false);
  }

  PMBuilder.addExtension(PassManagerBuilder::EP_ModuleOptimizerEarly,
                         addSwiftStackPromotionPass);

  // If the optimizer is enabled, we run the ARCOpt pass in the scalar optimizer
  // and the Contract pass as late as possible.
  if (!Opts.DisableLLVMARCOpts) {
    PMBuilder.addExtension(PassManagerBuilder::EP_ScalarOptimizerLate,
                           addSwiftARCOptPass);
    PMBuilder.addExtension(PassManagerBuilder::EP_OptimizerLast,
                           addSwiftContractPass);
  }
  
  // Configure the function passes.
  legacy::FunctionPassManager FunctionPasses(Module);
  FunctionPasses.add(createTargetTransformInfoWrapperPass(
      TargetMachine->getTargetIRAnalysis()));
  if (Opts.Verify)
    FunctionPasses.add(createVerifierPass());
  PMBuilder.populateFunctionPassManager(FunctionPasses);

  // The PMBuilder only knows about LLVM AA passes.  We should explicitly add
  // the swift AA pass after the other ones.
  if (!Opts.DisableLLVMARCOpts) {
    FunctionPasses.add(createSwiftAAWrapperPass());
    FunctionPasses.add(createExternalAAWrapperPass([](Pass &P, Function &,
                                                      AAResults &AAR) {
      if (auto *WrapperPass = P.getAnalysisIfAvailable<SwiftAAWrapperPass>())
        AAR.addAAResult(WrapperPass->getResult());
    }));
  }

  // Run the function passes.
  FunctionPasses.doInitialization();
  for (auto I = Module->begin(), E = Module->end(); I != E; ++I)
    if (!I->isDeclaration())
      FunctionPasses.run(*I);
  FunctionPasses.doFinalization();

  // Configure the module passes.
  legacy::PassManager ModulePasses;
  ModulePasses.add(createTargetTransformInfoWrapperPass(
      TargetMachine->getTargetIRAnalysis()));
  PMBuilder.populateModulePassManager(ModulePasses);

  // The PMBuilder only knows about LLVM AA passes.  We should explicitly add
  // the swift AA pass after the other ones.
  if (!Opts.DisableLLVMARCOpts) {
    ModulePasses.add(createSwiftAAWrapperPass());
    ModulePasses.add(createExternalAAWrapperPass([](Pass &P, Function &,
                                                    AAResults &AAR) {
      if (auto *WrapperPass = P.getAnalysisIfAvailable<SwiftAAWrapperPass>())
        AAR.addAAResult(WrapperPass->getResult());
    }));
  }

  // If we're generating a profile, add the lowering pass now.
  if (Opts.GenerateProfile)
    ModulePasses.add(createInstrProfilingPass());

  if (Opts.Verify)
    ModulePasses.add(createVerifierPass());

  // Do it.
  ModulePasses.run(*Module);
}
