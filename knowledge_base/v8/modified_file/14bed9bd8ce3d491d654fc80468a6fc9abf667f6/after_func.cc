void CodeGenerator::AssembleMove(InstructionOperand* source,
                                 InstructionOperand* destination) {
  X64OperandConverter g(this, nullptr);
  // Helper function to write the given constant to the dst register.
  auto MoveConstantToRegister = [&](Register dst, Constant src) {
    switch (src.type()) {
      case Constant::kInt32: {
        if (RelocInfo::IsWasmPtrReference(src.rmode())) {
          __ movq(dst, src.ToInt64(), src.rmode());
        } else {
          int32_t value = src.ToInt32();
          if (value == 0) {
            __ xorl(dst, dst);
          } else {
            __ movl(dst, Immediate(value));
          }
        }
        break;
      }
      case Constant::kInt64:
        if (RelocInfo::IsWasmPtrReference(src.rmode())) {
          __ movq(dst, src.ToInt64(), src.rmode());
        } else {
          __ Set(dst, src.ToInt64());
        }
        break;
      case Constant::kFloat32:
        __ MoveNumber(dst, src.ToFloat32());
        break;
      case Constant::kFloat64:
        __ MoveNumber(dst, src.ToFloat64().value());
        break;
      case Constant::kExternalReference:
        __ Move(dst, src.ToExternalReference());
        break;
      case Constant::kHeapObject: {
        Handle<HeapObject> src_object = src.ToHeapObject();
        Heap::RootListIndex index;
        if (IsMaterializableFromRoot(src_object, &index)) {
          __ LoadRoot(dst, index);
        } else {
          __ Move(dst, src_object);
        }
        break;
      }
      case Constant::kRpoNumber:
        UNREACHABLE();  // TODO(dcarney): load of labels on x64.
        break;
    }
  };
  // Helper function to write the given constant to the stack.
  auto MoveConstantToSlot = [&](Operand dst, Constant src) {
    if (!RelocInfo::IsWasmPtrReference(src.rmode())) {
      switch (src.type()) {
        case Constant::kInt32:
          __ movq(dst, Immediate(src.ToInt32()));
          return;
        case Constant::kInt64:
          __ Set(dst, src.ToInt64());
          return;
        default:
          break;
      }
    }
    MoveConstantToRegister(kScratchRegister, src);
    __ movq(dst, kScratchRegister);
  };
  // Dispatch on the source and destination operand kinds.
  switch (MoveType::InferMove(source, destination)) {
    case MoveType::kRegisterToRegister:
      if (source->IsRegister()) {
        __ movq(g.ToRegister(destination), g.ToRegister(source));
      } else {
        DCHECK(source->IsFPRegister());
        __ Movapd(g.ToDoubleRegister(destination), g.ToDoubleRegister(source));
      }
      return;
    case MoveType::kRegisterToStack: {
      Operand dst = g.ToOperand(destination);
      if (source->IsRegister()) {
        __ movq(dst, g.ToRegister(source));
      } else {
        DCHECK(source->IsFPRegister());
        XMMRegister src = g.ToDoubleRegister(source);
        MachineRepresentation rep =
            LocationOperand::cast(source)->representation();
        if (rep != MachineRepresentation::kSimd128) {
          __ Movsd(dst, src);
        } else {
          __ Movups(dst, src);
        }
      }
      return;
    }
    case MoveType::kStackToRegister: {
      Operand src = g.ToOperand(source);
      if (source->IsStackSlot()) {
        __ movq(g.ToRegister(destination), src);
      } else {
        DCHECK(source->IsFPStackSlot());
        XMMRegister dst = g.ToDoubleRegister(destination);
        MachineRepresentation rep =
            LocationOperand::cast(source)->representation();
        if (rep != MachineRepresentation::kSimd128) {
          __ Movsd(dst, src);
        } else {
          __ Movups(dst, src);
        }
      }
      return;
    }
    case MoveType::kStackToStack: {
      Operand src = g.ToOperand(source);
      Operand dst = g.ToOperand(destination);
      if (source->IsStackSlot()) {
        // Spill on demand to use a temporary register for memory-to-memory
        // moves.
        __ movq(kScratchRegister, src);
        __ movq(dst, kScratchRegister);
      } else {
        MachineRepresentation rep =
            LocationOperand::cast(source)->representation();
        if (rep != MachineRepresentation::kSimd128) {
          __ Movsd(kScratchDoubleReg, src);
          __ Movsd(dst, kScratchDoubleReg);
        } else {
          DCHECK(source->IsSimd128StackSlot());
          __ Movups(kScratchDoubleReg, src);
          __ Movups(dst, kScratchDoubleReg);
        }
      }
      return;
    }
    case MoveType::kConstantToRegister: {
      Constant src = g.ToConstant(source);
      if (destination->IsRegister()) {
        MoveConstantToRegister(g.ToRegister(destination), src);
      } else {
        DCHECK(destination->IsFPRegister());
        XMMRegister dst = g.ToDoubleRegister(destination);
        if (src.type() == Constant::kFloat32) {
          // TODO(turbofan): Can we do better here?
          __ Move(dst, bit_cast<uint32_t>(src.ToFloat32()));
        } else {
          DCHECK_EQ(src.type(), Constant::kFloat64);
          __ Move(dst, src.ToFloat64().AsUint64());
        }
      }
      return;
    }
    case MoveType::kConstantToStack: {
      Constant src = g.ToConstant(source);
      Operand dst = g.ToOperand(destination);
      if (destination->IsStackSlot()) {
        MoveConstantToSlot(dst, src);
      } else {
        DCHECK(destination->IsFPStackSlot());
        if (src.type() == Constant::kFloat32) {
          __ movl(dst, Immediate(bit_cast<uint32_t>(src.ToFloat32())));
        } else {
          DCHECK_EQ(src.type(), Constant::kFloat64);
          __ movq(kScratchRegister, src.ToFloat64().AsUint64());
          __ movq(dst, kScratchRegister);
        }
      }
      return;
    }
  }
  UNREACHABLE();
}
