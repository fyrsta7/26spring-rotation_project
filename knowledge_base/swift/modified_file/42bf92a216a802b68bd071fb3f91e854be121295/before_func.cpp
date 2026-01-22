InlineCost swift::instructionInlineCost(SILInstruction &I) {
  switch (I.getKind()) {
  case SILInstructionKind::IntegerLiteralInst:
  case SILInstructionKind::FloatLiteralInst:
  case SILInstructionKind::DebugValueInst:
  case SILInstructionKind::DebugValueAddrInst:
  case SILInstructionKind::StringLiteralInst:
  case SILInstructionKind::FixLifetimeInst:
  case SILInstructionKind::EndBorrowInst:
  case SILInstructionKind::BeginBorrowInst:
  case SILInstructionKind::MarkDependenceInst:
  case SILInstructionKind::PreviousDynamicFunctionRefInst:
  case SILInstructionKind::DynamicFunctionRefInst:
  case SILInstructionKind::FunctionRefInst:
  case SILInstructionKind::AllocGlobalInst:
  case SILInstructionKind::GlobalAddrInst:
  case SILInstructionKind::BaseAddrForOffsetInst:
  case SILInstructionKind::EndLifetimeInst:
  case SILInstructionKind::UncheckedOwnershipConversionInst:
    return InlineCost::Free;

  // Typed GEPs are free.
  case SILInstructionKind::TupleElementAddrInst:
  case SILInstructionKind::StructElementAddrInst:
  case SILInstructionKind::ProjectBlockStorageInst:
    return InlineCost::Free;

  // Aggregates are exploded at the IR level; these are effectively no-ops.
  case SILInstructionKind::TupleInst:
  case SILInstructionKind::StructInst:
  case SILInstructionKind::StructExtractInst:
  case SILInstructionKind::TupleExtractInst:
  case SILInstructionKind::DestructureStructInst:
  case SILInstructionKind::DestructureTupleInst:
    return InlineCost::Free;

  // Unchecked casts are free.
  case SILInstructionKind::AddressToPointerInst:
  case SILInstructionKind::PointerToAddressInst:

  case SILInstructionKind::UncheckedRefCastInst:
  case SILInstructionKind::UncheckedRefCastAddrInst:
  case SILInstructionKind::UncheckedAddrCastInst:
  case SILInstructionKind::UncheckedTrivialBitCastInst:
  case SILInstructionKind::UncheckedBitwiseCastInst:
  case SILInstructionKind::UncheckedValueCastInst:

  case SILInstructionKind::RawPointerToRefInst:
  case SILInstructionKind::RefToRawPointerInst:

  case SILInstructionKind::UpcastInst:

  case SILInstructionKind::ThinToThickFunctionInst:
  case SILInstructionKind::ThinFunctionToPointerInst:
  case SILInstructionKind::PointerToThinFunctionInst:
  case SILInstructionKind::ConvertFunctionInst:
  case SILInstructionKind::ConvertEscapeToNoEscapeInst:

  case SILInstructionKind::BridgeObjectToWordInst:
    return InlineCost::Free;

  // Access instructions are free unless we're dynamically enforcing them.
  case SILInstructionKind::BeginAccessInst:
    return getEnforcementCost(cast<BeginAccessInst>(I).getEnforcement());
  case SILInstructionKind::EndAccessInst:
    return getEnforcementCost(cast<EndAccessInst>(I).getBeginAccess()
                                                   ->getEnforcement());
  case SILInstructionKind::BeginUnpairedAccessInst:
    return getEnforcementCost(cast<BeginUnpairedAccessInst>(I)
                                .getEnforcement());
  case SILInstructionKind::EndUnpairedAccessInst:
    return getEnforcementCost(cast<EndUnpairedAccessInst>(I)
                                .getEnforcement());

  // TODO: These are free if the metatype is for a Swift class.
  case SILInstructionKind::ThickToObjCMetatypeInst:
  case SILInstructionKind::ObjCToThickMetatypeInst:
    return InlineCost::Expensive;
    
  // TODO: Bridge object conversions imply a masking operation that should be
  // "hella cheap" but not really expensive.
  case SILInstructionKind::BridgeObjectToRefInst:
  case SILInstructionKind::RefToBridgeObjectInst:
  case SILInstructionKind::ClassifyBridgeObjectInst:
  case SILInstructionKind::ValueToBridgeObjectInst:
    return InlineCost::Expensive;

  case SILInstructionKind::MetatypeInst:
    // Thin metatypes are always free.
    if (cast<MetatypeInst>(I).getType().castTo<MetatypeType>()
          ->getRepresentation() == MetatypeRepresentation::Thin)
      return InlineCost::Free;
    // TODO: Thick metatypes are free if they don't require generic or lazy
    // instantiation.
    return InlineCost::Expensive;

  // Protocol descriptor references are free.
  case SILInstructionKind::ObjCProtocolInst:
    return InlineCost::Free;

  // Metatype-to-object conversions are free.
  case SILInstructionKind::ObjCExistentialMetatypeToObjectInst:
  case SILInstructionKind::ObjCMetatypeToObjectInst:
    return InlineCost::Free;

  // Return and unreachable are free.
  case SILInstructionKind::UnreachableInst:
  case SILInstructionKind::ReturnInst:
  case SILInstructionKind::ThrowInst:
  case SILInstructionKind::UnwindInst:
  case SILInstructionKind::YieldInst:
  case SILInstructionKind::EndCOWMutationInst:
    return InlineCost::Free;

  case SILInstructionKind::AbortApplyInst:
  case SILInstructionKind::ApplyInst:
  case SILInstructionKind::TryApplyInst:
  case SILInstructionKind::AllocBoxInst:
  case SILInstructionKind::AllocExistentialBoxInst:
  case SILInstructionKind::AllocRefInst:
  case SILInstructionKind::AllocRefDynamicInst:
  case SILInstructionKind::AllocStackInst:
  case SILInstructionKind::AllocValueBufferInst:
  case SILInstructionKind::BindMemoryInst:
  case SILInstructionKind::BeginApplyInst:
  case SILInstructionKind::ValueMetatypeInst:
  case SILInstructionKind::WitnessMethodInst:
  case SILInstructionKind::AssignInst:
  case SILInstructionKind::AssignByWrapperInst:
  case SILInstructionKind::BranchInst:
  case SILInstructionKind::CheckedCastBranchInst:
  case SILInstructionKind::CheckedCastValueBranchInst:
  case SILInstructionKind::CheckedCastAddrBranchInst:
  case SILInstructionKind::ClassMethodInst:
  case SILInstructionKind::ObjCMethodInst:
  case SILInstructionKind::CondBranchInst:
  case SILInstructionKind::CondFailInst:
  case SILInstructionKind::CopyBlockInst:
  case SILInstructionKind::CopyBlockWithoutEscapingInst:
  case SILInstructionKind::CopyAddrInst:
  case SILInstructionKind::RetainValueInst:
  case SILInstructionKind::RetainValueAddrInst:
  case SILInstructionKind::UnmanagedRetainValueInst:
  case SILInstructionKind::CopyValueInst:
  case SILInstructionKind::DeallocBoxInst:
  case SILInstructionKind::DeallocExistentialBoxInst:
  case SILInstructionKind::DeallocRefInst:
  case SILInstructionKind::DeallocPartialRefInst:
  case SILInstructionKind::DeallocStackInst:
  case SILInstructionKind::DeallocValueBufferInst:
  case SILInstructionKind::DeinitExistentialAddrInst:
  case SILInstructionKind::DeinitExistentialValueInst:
  case SILInstructionKind::DestroyAddrInst:
  case SILInstructionKind::EndApplyInst:
  case SILInstructionKind::ProjectValueBufferInst:
  case SILInstructionKind::ProjectBoxInst:
  case SILInstructionKind::ProjectExistentialBoxInst:
  case SILInstructionKind::ReleaseValueInst:
  case SILInstructionKind::ReleaseValueAddrInst:
  case SILInstructionKind::UnmanagedReleaseValueInst:
  case SILInstructionKind::DestroyValueInst:
  case SILInstructionKind::AutoreleaseValueInst:
  case SILInstructionKind::UnmanagedAutoreleaseValueInst:
  case SILInstructionKind::DynamicMethodBranchInst:
  case SILInstructionKind::EnumInst:
  case SILInstructionKind::IndexAddrInst:
  case SILInstructionKind::TailAddrInst:
  case SILInstructionKind::IndexRawPointerInst:
  case SILInstructionKind::InitEnumDataAddrInst:
  case SILInstructionKind::InitExistentialAddrInst:
  case SILInstructionKind::InitExistentialValueInst:
  case SILInstructionKind::InitExistentialMetatypeInst:
  case SILInstructionKind::InitExistentialRefInst:
  case SILInstructionKind::InjectEnumAddrInst:
  case SILInstructionKind::LoadInst:
  case SILInstructionKind::LoadBorrowInst:
  case SILInstructionKind::OpenExistentialAddrInst:
  case SILInstructionKind::OpenExistentialBoxInst:
  case SILInstructionKind::OpenExistentialBoxValueInst:
  case SILInstructionKind::OpenExistentialMetatypeInst:
  case SILInstructionKind::OpenExistentialRefInst:
  case SILInstructionKind::OpenExistentialValueInst:
  case SILInstructionKind::PartialApplyInst:
  case SILInstructionKind::ExistentialMetatypeInst:
  case SILInstructionKind::RefElementAddrInst:
  case SILInstructionKind::RefTailAddrInst:
  case SILInstructionKind::StoreInst:
  case SILInstructionKind::StoreBorrowInst:
  case SILInstructionKind::StrongReleaseInst:
  case SILInstructionKind::SetDeallocatingInst:
  case SILInstructionKind::StrongRetainInst:
  case SILInstructionKind::SuperMethodInst:
  case SILInstructionKind::ObjCSuperMethodInst:
  case SILInstructionKind::SwitchEnumAddrInst:
  case SILInstructionKind::SwitchEnumInst:
  case SILInstructionKind::SwitchValueInst:
  case SILInstructionKind::UncheckedEnumDataInst:
  case SILInstructionKind::UncheckedTakeEnumDataAddrInst:
  case SILInstructionKind::UnconditionalCheckedCastInst:
  case SILInstructionKind::UnconditionalCheckedCastAddrInst:
  case SILInstructionKind::UnconditionalCheckedCastValueInst:
  case SILInstructionKind::IsEscapingClosureInst:
  case SILInstructionKind::IsUniqueInst:
  case SILInstructionKind::BeginCOWMutationInst:
  case SILInstructionKind::InitBlockStorageHeaderInst:
  case SILInstructionKind::SelectEnumAddrInst:
  case SILInstructionKind::SelectEnumInst:
  case SILInstructionKind::SelectValueInst:
  case SILInstructionKind::KeyPathInst:
  case SILInstructionKind::GlobalValueInst:
  case SILInstructionKind::DifferentiableFunctionInst:
  case SILInstructionKind::LinearFunctionInst:
  case SILInstructionKind::DifferentiableFunctionExtractInst:
  case SILInstructionKind::LinearFunctionExtractInst:
  case SILInstructionKind::DifferentiabilityWitnessFunctionInst:
#define COMMON_ALWAYS_OR_SOMETIMES_LOADABLE_CHECKED_REF_STORAGE(Name)          \
  case SILInstructionKind::Name##ToRefInst:                                    \
  case SILInstructionKind::RefTo##Name##Inst:                                  \
  case SILInstructionKind::StrongCopy##Name##ValueInst:
#define NEVER_LOADABLE_CHECKED_REF_STORAGE(Name, ...) \
  case SILInstructionKind::Load##Name##Inst: \
  case SILInstructionKind::Store##Name##Inst:
#define ALWAYS_LOADABLE_CHECKED_REF_STORAGE(Name, ...)                         \
  COMMON_ALWAYS_OR_SOMETIMES_LOADABLE_CHECKED_REF_STORAGE(Name)                \
  case SILInstructionKind::Name##RetainInst:                                   \
  case SILInstructionKind::Name##ReleaseInst:                                  \
  case SILInstructionKind::StrongRetain##Name##Inst:
#define SOMETIMES_LOADABLE_CHECKED_REF_STORAGE(Name, ...) \
  NEVER_LOADABLE_CHECKED_REF_STORAGE(Name, "...") \
  ALWAYS_LOADABLE_CHECKED_REF_STORAGE(Name, "...")
#define UNCHECKED_REF_STORAGE(Name, ...) \
  COMMON_ALWAYS_OR_SOMETIMES_LOADABLE_CHECKED_REF_STORAGE(Name)
#include "swift/AST/ReferenceStorage.def"
#undef COMMON_ALWAYS_OR_SOMETIMES_LOADABLE_CHECKED_REF_STORAGE
    return InlineCost::Expensive;

  case SILInstructionKind::BuiltinInst: {
    auto *BI = cast<BuiltinInst>(&I);
    // Expect intrinsics are 'free' instructions.
    if (BI->getIntrinsicInfo().ID == llvm::Intrinsic::expect)
      return InlineCost::Free;
    if (BI->getBuiltinInfo().ID == BuiltinValueKind::OnFastPath)
      return InlineCost::Free;

    return InlineCost::Expensive;
  }
  case SILInstructionKind::MarkFunctionEscapeInst:
  case SILInstructionKind::MarkUninitializedInst:
    llvm_unreachable("not valid in canonical sil");
  case SILInstructionKind::ObjectInst:
    llvm_unreachable("not valid in a function");
  }

  llvm_unreachable("Unhandled ValueKind in switch.");
}
