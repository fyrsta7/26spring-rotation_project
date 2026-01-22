InlineCost swift::instructionInlineCost(SILInstruction &I) {
  switch (I.getKind()) {
    case ValueKind::IntegerLiteralInst:
    case ValueKind::FloatLiteralInst:
    case ValueKind::DebugValueInst:
    case ValueKind::DebugValueAddrInst:
    case ValueKind::StringLiteralInst:
    case ValueKind::ConstStringLiteralInst:
    case ValueKind::FixLifetimeInst:
    case ValueKind::EndBorrowInst:
    case ValueKind::EndBorrowArgumentInst:
    case ValueKind::BeginBorrowInst:
    case ValueKind::MarkDependenceInst:
    case ValueKind::FunctionRefInst:
    case ValueKind::AllocGlobalInst:
    case ValueKind::GlobalAddrInst:
    case ValueKind::EndLifetimeInst:
    case ValueKind::UncheckedOwnershipConversionInst:
      return InlineCost::Free;

    // Typed GEPs are free.
    case ValueKind::TupleElementAddrInst:
    case ValueKind::StructElementAddrInst:
    case ValueKind::ProjectBlockStorageInst:
      return InlineCost::Free;

    // Aggregates are exploded at the IR level; these are effectively no-ops.
    case ValueKind::TupleInst:
    case ValueKind::StructInst:
    case ValueKind::StructExtractInst:
    case ValueKind::TupleExtractInst:
      return InlineCost::Free;

    // Unchecked casts are free.
    case ValueKind::AddressToPointerInst:
    case ValueKind::PointerToAddressInst:

    case ValueKind::UncheckedRefCastInst:
    case ValueKind::UncheckedRefCastAddrInst:
    case ValueKind::UncheckedAddrCastInst:
    case ValueKind::UncheckedTrivialBitCastInst:
    case ValueKind::UncheckedBitwiseCastInst:

    case ValueKind::RawPointerToRefInst:
    case ValueKind::RefToRawPointerInst:

    case ValueKind::UpcastInst:

    case ValueKind::ThinToThickFunctionInst:
    case ValueKind::ThinFunctionToPointerInst:
    case ValueKind::PointerToThinFunctionInst:
    case ValueKind::ConvertFunctionInst:

    case ValueKind::BridgeObjectToWordInst:
      return InlineCost::Free;

    // Access instructions are free unless we're dynamically enforcing them.
    case ValueKind::BeginAccessInst:
      return getEnforcementCost(cast<BeginAccessInst>(I));
    case ValueKind::EndAccessInst:
      return getEnforcementCost(*cast<EndAccessInst>(I).getBeginAccess());

    // TODO: These are free if the metatype is for a Swift class.
    case ValueKind::ThickToObjCMetatypeInst:
    case ValueKind::ObjCToThickMetatypeInst:
      return InlineCost::Expensive;
      
    // TODO: Bridge object conversions imply a masking operation that should be
    // "hella cheap" but not really expensive
    case ValueKind::BridgeObjectToRefInst:
    case ValueKind::RefToBridgeObjectInst:
      return InlineCost::Expensive;

    case ValueKind::MetatypeInst:
      // Thin metatypes are always free.
      if (I.getType().castTo<MetatypeType>()->getRepresentation()
            == MetatypeRepresentation::Thin)
        return InlineCost::Free;
      // TODO: Thick metatypes are free if they don't require generic or lazy
      // instantiation.
      return InlineCost::Expensive;

    // Protocol descriptor references are free.
    case ValueKind::ObjCProtocolInst:
      return InlineCost::Free;

    // Metatype-to-object conversions are free.
    case ValueKind::ObjCExistentialMetatypeToObjectInst:
    case ValueKind::ObjCMetatypeToObjectInst:
      return InlineCost::Free;

    // Return and unreachable are free.
    case ValueKind::UnreachableInst:
    case ValueKind::ReturnInst:
    case ValueKind::ThrowInst:
      return InlineCost::Free;

    case ValueKind::ApplyInst:
    case ValueKind::TryApplyInst:
    case ValueKind::AllocBoxInst:
    case ValueKind::AllocExistentialBoxInst:
    case ValueKind::AllocRefInst:
    case ValueKind::AllocRefDynamicInst:
    case ValueKind::AllocStackInst:
    case ValueKind::AllocValueBufferInst:
    case ValueKind::BindMemoryInst:
    case ValueKind::ValueMetatypeInst:
    case ValueKind::WitnessMethodInst:
    case ValueKind::AssignInst:
    case ValueKind::BranchInst:
    case ValueKind::CheckedCastBranchInst:
    case ValueKind::CheckedCastValueBranchInst:
    case ValueKind::CheckedCastAddrBranchInst:
    case ValueKind::ClassMethodInst:
    case ValueKind::CondBranchInst:
    case ValueKind::CondFailInst:
    case ValueKind::CopyBlockInst:
    case ValueKind::CopyAddrInst:
    case ValueKind::RetainValueInst:
    case ValueKind::UnmanagedRetainValueInst:
    case ValueKind::CopyValueInst:
    case ValueKind::CopyUnownedValueInst:
    case ValueKind::DeallocBoxInst:
    case ValueKind::DeallocExistentialBoxInst:
    case ValueKind::DeallocRefInst:
    case ValueKind::DeallocPartialRefInst:
    case ValueKind::DeallocStackInst:
    case ValueKind::DeallocValueBufferInst:
    case ValueKind::DeinitExistentialAddrInst:
    case ValueKind::DeinitExistentialOpaqueInst:
    case ValueKind::DestroyAddrInst:
    case ValueKind::ProjectValueBufferInst:
    case ValueKind::ProjectBoxInst:
    case ValueKind::ProjectExistentialBoxInst:
    case ValueKind::ReleaseValueInst:
    case ValueKind::UnmanagedReleaseValueInst:
    case ValueKind::DestroyValueInst:
    case ValueKind::AutoreleaseValueInst:
    case ValueKind::UnmanagedAutoreleaseValueInst:
    case ValueKind::DynamicMethodBranchInst:
    case ValueKind::DynamicMethodInst:
    case ValueKind::EnumInst:
    case ValueKind::IndexAddrInst:
    case ValueKind::TailAddrInst:
    case ValueKind::IndexRawPointerInst:
    case ValueKind::InitEnumDataAddrInst:
    case ValueKind::InitExistentialAddrInst:
    case ValueKind::InitExistentialOpaqueInst:
    case ValueKind::InitExistentialMetatypeInst:
    case ValueKind::InitExistentialRefInst:
    case ValueKind::InjectEnumAddrInst:
    case ValueKind::IsNonnullInst:
    case ValueKind::LoadInst:
    case ValueKind::LoadBorrowInst:
    case ValueKind::LoadUnownedInst:
    case ValueKind::LoadWeakInst:
    case ValueKind::OpenExistentialAddrInst:
    case ValueKind::OpenExistentialBoxInst:
    case ValueKind::OpenExistentialMetatypeInst:
    case ValueKind::OpenExistentialRefInst:
    case ValueKind::OpenExistentialOpaqueInst:
    case ValueKind::PartialApplyInst:
    case ValueKind::ExistentialMetatypeInst:
    case ValueKind::RefElementAddrInst:
    case ValueKind::RefTailAddrInst:
    case ValueKind::RefToUnmanagedInst:
    case ValueKind::RefToUnownedInst:
    case ValueKind::StoreInst:
    case ValueKind::StoreBorrowInst:
    case ValueKind::StoreUnownedInst:
    case ValueKind::StoreWeakInst:
    case ValueKind::StrongPinInst:
    case ValueKind::StrongReleaseInst:
    case ValueKind::SetDeallocatingInst:
    case ValueKind::StrongRetainInst:
    case ValueKind::StrongRetainUnownedInst:
    case ValueKind::StrongUnpinInst:
    case ValueKind::SuperMethodInst:
    case ValueKind::SwitchEnumAddrInst:
    case ValueKind::SwitchEnumInst:
    case ValueKind::SwitchValueInst:
    case ValueKind::UncheckedEnumDataInst:
    case ValueKind::UncheckedTakeEnumDataAddrInst:
    case ValueKind::UnconditionalCheckedCastInst:
    case ValueKind::UnconditionalCheckedCastAddrInst:
    case ValueKind::UnconditionalCheckedCastValueInst:
    case ValueKind::UnmanagedToRefInst:
    case ValueKind::UnownedReleaseInst:
    case ValueKind::UnownedRetainInst:
    case ValueKind::IsUniqueInst:
    case ValueKind::IsUniqueOrPinnedInst:
    case ValueKind::UnownedToRefInst:
    case ValueKind::InitBlockStorageHeaderInst:
    case ValueKind::SelectEnumAddrInst:
    case ValueKind::SelectEnumInst:
    case ValueKind::SelectValueInst:
      return InlineCost::Expensive;

    case ValueKind::BuiltinInst: {
      auto *BI = cast<BuiltinInst>(&I);
      // Expect intrinsics are 'free' instructions.
      if (BI->getIntrinsicInfo().ID == llvm::Intrinsic::expect)
        return InlineCost::Free;
      if (BI->getBuiltinInfo().ID == BuiltinValueKind::OnFastPath)
        return InlineCost::Free;

      return InlineCost::Expensive;
    }
    case ValueKind::SILPHIArgument:
    case ValueKind::SILFunctionArgument:
    case ValueKind::SILUndef:
      llvm_unreachable("Only instructions should be passed into this "
                       "function.");
    case ValueKind::MarkFunctionEscapeInst:
    case ValueKind::MarkUninitializedInst:
    case ValueKind::MarkUninitializedBehaviorInst:
      llvm_unreachable("not valid in canonical sil");
  }

  llvm_unreachable("Unhandled ValueKind in switch.");
}
