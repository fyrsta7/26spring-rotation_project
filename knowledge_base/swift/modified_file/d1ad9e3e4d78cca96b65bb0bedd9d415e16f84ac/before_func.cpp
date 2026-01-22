static void lowerAssignInstruction(SILBuilderWithScope &b, AssignInst *inst) {
  LLVM_DEBUG(llvm::dbgs() << "  *** Lowering [isInit="
                          << unsigned(inst->getOwnershipQualifier())
                          << "]: " << *inst << "\n");

  ++numAssignRewritten;

  SILValue src = inst->getSrc();
  SILValue dest = inst->getDest();
  SILLocation loc = inst->getLoc();
  AssignOwnershipQualifier qualifier = inst->getOwnershipQualifier();

  // Unknown qualifier is considered unprocessed. Just lower it as [reassign],
  // but if the destination type is trivial, treat it as [init].
  //
  // Unknown should not be lowered because definite initialization should
  // always set an initialization kind for assign instructions, but there exists
  // some situations where SILGen doesn't generate a mark_uninitialized
  // instruction for a full mark_uninitialized. Thus definite initialization
  // doesn't set an initialization kind for some assign instructions.
  //
  // TODO: Fix SILGen so that this is an assert preventing the lowering of
  //       Unknown init kind.
  if (qualifier == AssignOwnershipQualifier::Unknown)
    qualifier = AssignOwnershipQualifier::Reassign;

  if (qualifier == AssignOwnershipQualifier::Init ||
      inst->getDest()->getType().isTrivial(*inst->getFunction())) {

    // If this is an initialization, or the storage type is trivial, we
    // can just replace the assignment with a store.
    assert(qualifier != AssignOwnershipQualifier::Reinit);
    b.createTrivialStoreOr(loc, src, dest, StoreOwnershipQualifier::Init);
    inst->eraseFromParent();
    return;
  }

  if (qualifier == AssignOwnershipQualifier::Reinit) {
    // We have a case where a convenience initializer on a class
    // delegates to a factory initializer from a protocol extension.
    // Factory initializers give us a whole new instance, so the existing
    // instance, which has not been initialized and never will be, must be
    // freed using dealloc_partial_ref.
    SILValue pointer = b.createLoad(loc, dest, LoadOwnershipQualifier::Take);
    b.createStore(loc, src, dest, StoreOwnershipQualifier::Init);

    auto metatypeTy = CanMetatypeType::get(
        dest->getType().getASTType(), MetatypeRepresentation::Thick);
    auto silMetatypeTy = SILType::getPrimitiveObjectType(metatypeTy);
    SILValue metatype = b.createValueMetatype(loc, silMetatypeTy, pointer);

    b.createDeallocPartialRef(loc, pointer, metatype);
    inst->eraseFromParent();
    return;
  }

  assert(qualifier == AssignOwnershipQualifier::Reassign);
  // Otherwise, we need to replace the assignment with a store [assign] which
  // lowers to the load/store/release dance. Note that the new value is already
  // considered to be retained (by the semantics of the storage type),
  // and we're transferring that ownership count into the destination.

  b.createStore(loc, src, dest, StoreOwnershipQualifier::Assign);
  inst->eraseFromParent();
}
