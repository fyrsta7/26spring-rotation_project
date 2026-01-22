SILInstruction *
CastOptimizer::
optimizeBridgedObjCToSwiftCast(SILInstruction *Inst,
                     bool isConditional,
                     SILValue Src,
                     SILValue Dest,
                     CanType Source,
                     CanType Target,
                     Type BridgedSourceTy,
                     Type BridgedTargetTy,
                     SILBasicBlock *SuccessBB,
                     SILBasicBlock *FailureBB) {
  auto &M = Inst->getModule();
  auto Loc = Inst->getLoc();

  CanType CanBridgedTy(BridgedTargetTy);
  SILType SILBridgedTy = SILType::getPrimitiveObjectType(CanBridgedTy);

  SILBuilderWithScope<1> Builder(Inst);
  SILValue SrcOp;
  SILInstruction *NewI = nullptr;

  assert(Src.getType().isAddress() && "Source should have an address type");
  assert(Dest.getType().isAddress() && "Source should have an address type");

  if (SILBridgedTy != Src.getType()) {
    // Check if we can simplify a cast into:
    // - ObjCTy to _ObjectiveCBridgeable._ObjectiveCType.
    // - then convert _ObjectiveCBridgeable._ObjectiveCType to
    // a Swift type using _forceBridgeFromObjectiveC.

    // Generate a load for the source argument.
    auto *Load = Builder.createLoad(Loc, Src);
    // Try to convert the source into the expected ObjC type first.
    // TODO: If type of the source and the expected ObjC type are
    // equal, there is no need to generate the conversion.
    if (isConditional) {
      SILBasicBlock *CastSuccessBB = Inst->getFunction()->createBasicBlock();
      CastSuccessBB->createBBArg(SILBridgedTy);
      NewI = Builder.createCheckedCastBranch(Loc, false, SILValue(Load, 0),
                                             SILBridgedTy, CastSuccessBB,
                                             FailureBB);
      Builder.setInsertionPoint(CastSuccessBB);
      SrcOp = SILValue(CastSuccessBB->getBBArg(0), 0);
    } else {
      NewI = Builder.createUnconditionalCheckedCast(Loc, SILValue(Load, 0),
                                                    SILBridgedTy);
      SrcOp = SILValue(NewI, 0);
    }
  } else {
    SrcOp = Src;
  }

  // Now emit the a cast from the casted ObjC object into a target type.
  // This is done by means of calling _forceBridgeFromObjectiveC or
  // _conditionallyBridgeFromObjectiveC_birdgeable from the Target type.
  // Lookup the required function in the Target type.

  // Lookup the _ObjectiveCBridgeable protocol.
  auto BridgedProto =
      M.getASTContext().getProtocol(KnownProtocolKind::_ObjectiveCBridgeable);
  auto Conf =
      M.getSwiftModule()->lookupConformance(Target, BridgedProto, nullptr);
  assert(Conf.getInt() == ConformanceKind::Conforms &&
         "_ObjectiveCBridgeable conformance should exist");

  auto *Conformance = Conf.getPointer();

  // The conformance to _BridgedToObjectiveC is statically known.
  // Retrieve the  bridging operation to be used if a static conformance
  // to _BridgedToObjectiveC can be proven.
  FuncDecl *BridgeFuncDecl =
      isConditional
          ? M.getASTContext().getConditionallyBridgeFromObjectiveCBridgeable(nullptr)
          : M.getASTContext().getForceBridgeFromObjectiveCBridgeable(nullptr);

  assert(BridgeFuncDecl && "_forceBridgeFromObjectiveC should exist");

  SILDeclRef FuncDeclRef(BridgeFuncDecl, SILDeclRef::Kind::Func);

  // Lookup a function from the stdlib.
  SILFunction *BridgedFunc = M.getOrCreateFunction(
      Loc, FuncDeclRef, ForDefinition_t::NotForDefinition);

  assert(BridgedFunc && "Bridging function was not found");

  auto *FuncRef = Builder.createFunctionRef(Loc, BridgedFunc);

  auto MetaTy = MetatypeType::get(Target, MetatypeRepresentation::Thick);
  auto SILMetaTy = M.Types.getTypeLowering(MetaTy, 0).getLoweredType();
  auto *MetaTyVal = Builder.createMetatype(Loc, SILMetaTy);
  SmallVector<SILValue, 1> Args;

  auto PolyFuncTy = BridgeFuncDecl->getType()->getAs<PolymorphicFunctionType>();
  ArrayRef<ArchetypeType *> Archetypes =
      PolyFuncTy->getGenericParams().getAllArchetypes();

  // Add substitutions
  SmallVector<Substitution, 2> Subs;
  auto Conformances = M.getASTContext().Allocate<ProtocolConformance *>(1);
  Conformances[0] = Conformance;
  Subs.push_back(Substitution(Archetypes[0], Target, Conformances));
  const Substitution *DepTypeSubst = getTypeWitnessByName(
      Conformance, M.getASTContext().getIdentifier("_ObjectiveCType"));
  Subs.push_back(Substitution(Archetypes[1], DepTypeSubst->getReplacement(),
                              DepTypeSubst->getConformances()));
  auto SILFnTy = FuncRef->getType();
  SILType SubstFnTy = SILFnTy.substGenericArgs(M, Subs);
  SILType ResultTy = SubstFnTy.castTo<SILFunctionType>()->getSILResult();

  // Temporary to hold the intermediate result.
  AllocStackInst *Tmp = nullptr;
  CanType OptionalTy;
  OptionalTypeKind OTK;
  SILValue InOutOptionalParam;
  if (isConditional) {
    // Create a temporary
    OptionalTy = OptionalType::get(Dest.getType().getSwiftRValueType())
                     ->getImplementationType()
                     .getCanonicalTypeOrNull();
    OptionalTy.getAnyOptionalObjectType(OTK);
    Tmp = Builder.createAllocStack(Loc,
                                   SILType::getPrimitiveObjectType(OptionalTy));
    InOutOptionalParam = SILValue(Tmp, 1);
  } else {
    InOutOptionalParam = Dest;
  }

  Args.push_back(InOutOptionalParam);
  Args.push_back(SrcOp);
  Args.push_back(SILValue(MetaTyVal, 0));

  auto *AI = Builder.createApply(Loc, FuncRef, SubstFnTy, ResultTy, Subs, Args);
  if (isConditional) {
    // Copy the temporary into Dest.
    // Load from the optional
    auto *SomeDecl = Builder.getASTContext().getOptionalSomeDecl(OTK);
    bool isNotAddressOnly = !InOutOptionalParam.getType().isTrivial(M) &&
                            !InOutOptionalParam.getType().isAddressOnly(M);
    auto Addr = Builder.createUncheckedTakeEnumDataAddr(Loc, InOutOptionalParam,
                                                        SomeDecl);
    auto LoadFromOptional = Builder.createLoad(Loc, SILValue(Addr, 0));
    if (isNotAddressOnly)
      Builder.createRetainValue(Loc, LoadFromOptional);
    // Store into Dest
    Builder.createStore(Loc, LoadFromOptional, Dest);
    if (isNotAddressOnly)
      Builder.createReleaseValue(Loc, LoadFromOptional);
    Builder.createDeallocStack(Loc, SILValue(Tmp, 0));
    SmallVector<SILValue, 1> SuccessBBArgs;
    Builder.createBranch(Loc, SuccessBB, SuccessBBArgs);
  }

  EraseInstAction(Inst);
  return (NewI) ? NewI : AI;
}
