void swift::irgen::emitClosure(IRGenFunction &IGF, CapturingExpr *E,
                               Explosion &explosion) {
  assert(isa<FuncExpr>(E) || isa<ClosureExpr>(E));

  ArrayRef<Pattern*> Patterns;
  if (FuncExpr *FE = dyn_cast<FuncExpr>(E))
    Patterns = FE->getParamPatterns();
  else
    Patterns = cast<ClosureExpr>(E)->getParamPatterns();

  if (Patterns.size() != 1) {
    IGF.unimplemented(E->getLoc(), "curried local functions");
    return IGF.emitFakeExplosion(IGF.getFragileTypeInfo(E->getType()),
                                 explosion);
  }

  for (ValueDecl *D : E->getCaptures()) {
    if (!isa<VarDecl>(D) && !isa<FuncDecl>(D)) {
      IGF.unimplemented(E->getLoc(), "capturing non-variables");
      return IGF.emitFakeExplosion(IGF.getFragileTypeInfo(E->getType()),
                                  explosion);
    }
  }

  bool HasCaptures = !E->getCaptures().empty();

  // Create the IR function.
  llvm::FunctionType *fnType =
      IGF.IGM.getFunctionType(E->getType(), ExplosionKind::Minimal, 0,
                              HasCaptures);
  llvm::Function *fn =
      llvm::Function::Create(fnType, llvm::GlobalValue::InternalLinkage,
                             "closure", &IGF.IGM.Module);

  IRGenFunction innerIGF(IGF.IGM, E->getType(), Patterns,
                         ExplosionKind::Minimal, /*uncurry level*/ 0, fn,
                         HasCaptures ? Prologue::StandardWithContext :
                                       Prologue::Standard);

  ManagedValue contextPtr(IGF.IGM.RefCountedNull);

  // There are two places we need to generate code for captures: in the
  // current function, to store the captures to a capture block, and in the
  // inner function, to load the captures from the capture block.
  if (HasCaptures) {
    SmallVector<const TypeInfo *, 4> Fields;
    for (ValueDecl *D : E->getCaptures()) {
      Type RefTy = D->getTypeOfReference();
      const TypeInfo &typeInfo = IGF.getFragileTypeInfo(RefTy);
      Fields.push_back(&typeInfo);
    }
    HeapLayout layout(IGF.IGM, LayoutStrategy::Optimal, Fields);

    // Allocate the capture block.
    contextPtr = IGF.emitAlloc(layout, "closure-data.alloc");
    
    Address CaptureStruct =
      layout.emitCastOfAlloc(IGF, contextPtr.getValue(), "closure-data");
    Address InnerStruct =
      layout.emitCastOfAlloc(innerIGF, innerIGF.ContextPtr, "closure-data");

    // Emit stores and loads for capture block
    for (unsigned i = 0, e = E->getCaptures().size(); i != e; ++i) {
      // FIXME: avoid capturing owner when this is obviously derivable.

      ValueDecl *D = E->getCaptures()[i];
      auto &elt = layout.getElements()[i];

      if (isa<FuncDecl>(D)) {
        Explosion OuterExplosion(ExplosionKind::Maximal);
        Address Func = IGF.getLocalFunc(cast<FuncDecl>(D));
        elt.Type->load(IGF, Func, OuterExplosion);
        Address CaptureAddr = elt.project(IGF, CaptureStruct);
        elt.Type->initialize(IGF, OuterExplosion, CaptureAddr);

        Address InnerAddr = elt.project(innerIGF, InnerStruct);
        innerIGF.setLocalFunc(cast<FuncDecl>(D), InnerAddr);
        continue;
      }

      Explosion OuterExplosion(ExplosionKind::Maximal);
      OwnedAddress Var = IGF.getLocalVar(cast<VarDecl>(D));
      IGF.emitLValueAsScalar(IGF.emitAddressLValue(Var),
                             OnHeap, OuterExplosion);
      Address CaptureAddr = elt.project(IGF, CaptureStruct);
      elt.Type->initialize(IGF, OuterExplosion, CaptureAddr);

      Address InnerAddr = elt.project(innerIGF, InnerStruct);
      Address InnerValueAddr =
        innerIGF.Builder.CreateStructGEP(InnerAddr, 0, Size(0));
      Address InnerOwnerAddr =
        innerIGF.Builder.CreateStructGEP(InnerAddr, 1, IGF.IGM.getPointerSize());
      Address InnerValue(innerIGF.Builder.CreateLoad(InnerValueAddr),
                         Var.getAddress().getAlignment());
      OwnedAddress InnerLocal(InnerValue,
                              innerIGF.Builder.CreateLoad(InnerOwnerAddr));
      innerIGF.setLocalVar(cast<VarDecl>(D), InnerLocal);
    }
  }

  if (FuncExpr *FE = dyn_cast<FuncExpr>(E)) {
    innerIGF.emitFunctionTopLevel(FE->getBody());
  } else {
    // Emit the body of the closure as if it were a single return
    // statement.
    ReturnStmt ret(SourceLoc(), cast<ClosureExpr>(E)->getBody());
    innerIGF.emitStmt(&ret);
  }

  // Build the explosion result.
  explosion.addUnmanaged(IGF.Builder.CreateBitCast(fn, IGF.IGM.Int8PtrTy));
  explosion.add(contextPtr);
}
