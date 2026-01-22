static InlineCost instructionInlineCost(SILInstruction &I) {
  switch (I.getKind()) {
    case ValueKind::FunctionRefInst:
    case ValueKind::BuiltinFunctionRefInst:
    case ValueKind::GlobalAddrInst:
    case ValueKind::SILGlobalAddrInst:
    case ValueKind::IntegerLiteralInst:
    case ValueKind::FloatLiteralInst:
    case ValueKind::DebugValueInst:
    case ValueKind::DebugValueAddrInst:
    case ValueKind::StringLiteralInst:
      return InlineCost::Free;
      
    case ValueKind::TupleElementAddrInst:
    case ValueKind::StructElementAddrInst: {
      // A gep whose operand is a gep with no other users will get folded by
      // LLVM into one gep implying the second should be free.
      SILValue Op = I.getOperand(0);
      if ((Op->getKind() == ValueKind::TupleElementAddrInst ||
           Op->getKind() == ValueKind::StructElementAddrInst) &&
          Op->hasOneUse())
        return InlineCost::Free;
    }
    // Aggregates are exploded at the IR level; these are effectively no-ops.
    case ValueKind::TupleInst:
    case ValueKind::StructInst:
    case ValueKind::StructExtractInst:
    case ValueKind::TupleExtractInst:
      return InlineCost::Free;
      
    // Unchecked casts are free.
    case ValueKind::AddressToPointerInst:
    case ValueKind::PointerToAddressInst:

    case ValueKind::ObjectPointerToRefInst:
    case ValueKind::RefToObjectPointerInst:
    
    case ValueKind::RawPointerToRefInst:
    case ValueKind::RefToRawPointerInst:
    
    case ValueKind::UpcastExistentialRefInst:
    case ValueKind::UpcastInst:
      
    case ValueKind::ThinToThickFunctionInst:
    case ValueKind::ConvertFunctionInst:
      return InlineCost::Free;
    
    case ValueKind::MetatypeInst:
      // Thin metatypes are always free.
      if (I.getType(0).castTo<MetatypeType>()->isThin())
        return InlineCost::Free;
      // TODO: Thick metatypes are free if they don't require generic or lazy
      // instantiation.
      return InlineCost::Expensive;

    // Return and unreachable are free.
    case ValueKind::UnreachableInst:
    case ValueKind::ReturnInst:
      return InlineCost::Free;
      
    // TODO
    case ValueKind::AllocArrayInst:
    case ValueKind::AllocBoxInst:
    case ValueKind::AllocRefInst:
    case ValueKind::AllocStackInst:
    case ValueKind::ApplyInst:
    case ValueKind::ArchetypeMetatypeInst:
    case ValueKind::ArchetypeMethodInst:
    case ValueKind::ArchetypeRefToSuperInst:
    case ValueKind::AssignInst:
    case ValueKind::AutoreleaseReturnInst:
    case ValueKind::BranchInst:
    case ValueKind::BridgeToBlockInst:
    case ValueKind::CheckedCastBranchInst:
    case ValueKind::ClassMetatypeInst:
    case ValueKind::ClassMethodInst:
    case ValueKind::CondBranchInst:
    case ValueKind::CondFailInst:
    case ValueKind::CopyAddrInst:
    case ValueKind::CopyValueInst:
    case ValueKind::DeallocBoxInst:
    case ValueKind::DeallocRefInst:
    case ValueKind::DeallocStackInst:
    case ValueKind::DeinitExistentialInst:
    case ValueKind::DestroyAddrInst:
    case ValueKind::DestroyValueInst:
    case ValueKind::DynamicMethodBranchInst:
    case ValueKind::DynamicMethodInst:
    case ValueKind::EnumInst:
    case ValueKind::IndexAddrInst:
    case ValueKind::IndexRawPointerInst:
    case ValueKind::InitEnumDataAddrInst:
    case ValueKind::InitExistentialInst:
    case ValueKind::InitExistentialRefInst:
    case ValueKind::InjectEnumAddrInst:
    case ValueKind::IsNonnullInst:
    case ValueKind::LoadInst:
    case ValueKind::LoadWeakInst:
    case ValueKind::PartialApplyInst:
    case ValueKind::PeerMethodInst:
    case ValueKind::ProjectExistentialInst:
    case ValueKind::ProjectExistentialRefInst:
    case ValueKind::ProtocolMetatypeInst:
    case ValueKind::ProtocolMethodInst:
    case ValueKind::RefElementAddrInst:
    case ValueKind::RefToUnownedInst:
    case ValueKind::StoreInst:
    case ValueKind::StoreWeakInst:
    case ValueKind::StrongReleaseInst:
    case ValueKind::StrongRetainAutoreleasedInst:
    case ValueKind::StrongRetainInst:
    case ValueKind::StrongRetainUnownedInst:
    case ValueKind::SuperMethodInst:
    case ValueKind::SwitchEnumAddrInst:
    case ValueKind::SwitchEnumInst:
    case ValueKind::SwitchIntInst:
    case ValueKind::TakeEnumDataAddrInst:
    case ValueKind::UnconditionalCheckedCastInst:
    case ValueKind::UnownedReleaseInst:
    case ValueKind::UnownedRetainInst:
    case ValueKind::UnownedToRefInst:
    case ValueKind::UpcastExistentialInst:
      return InlineCost::Expensive;

    case ValueKind::SILArgument:
    case ValueKind::SILUndef:
      llvm_unreachable("Only instructions should be passed into this "
                       "function.");    
    case ValueKind::MarkFunctionEscapeInst:
    case ValueKind::MarkUninitializedInst:
      llvm_unreachable("not valid in canonical sil");
  }
}
