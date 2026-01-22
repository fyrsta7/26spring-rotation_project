          InstructionCode opcode =
              kX64Cmp32 | AddressingModeField::encode(kMode_Root);
          return VisitCompare(
              selector, opcode,
              g.TempImmediate(
                  MacroAssemblerBase::RootRegisterOffsetForRootIndex(
                      root_index)),
              g.UseRegister(left), cont);
        }
      }
    }
  }
  VisitWordCompare(selector, node, kX64Cmp32, cont);
}

void VisitCompareZero(InstructionSelectorT<TurboshaftAdapter>* selector,
                      turboshaft::OpIndex user, turboshaft::OpIndex node,
                      InstructionCode opcode,
                      FlagsContinuationT<TurboshaftAdapter>* cont) {
  using namespace turboshaft;  // NOLINT(build/namespaces)
  X64OperandGeneratorT<TurboshaftAdapter> g(selector);
  const Operation& op = selector->turboshaft_graph()->Get(node);
  if (cont->IsBranch() &&
      (cont->condition() == kNotEqual || cont->condition() == kEqual)) {
    if (const WordBinopOp* binop = op.TryCast<WordBinopOp>()) {
      if (selector->IsOnlyUserOfNodeInSameBlock(user, node)) {
        const bool is64 = binop->rep == WordRepresentation::Word64();
        switch (binop->kind) {
          case WordBinopOp::Kind::kAdd:
            return VisitBinop(selector, node, is64 ? kX64Add : kX64Add32, cont);
          case WordBinopOp::Kind::kSub:
            return VisitBinop(selector, node, is64 ? kX64Sub : kX64Sub32, cont);
          case WordBinopOp::Kind::kBitwiseAnd:
            return VisitBinop(selector, node, is64 ? kX64And : kX64And32, cont);
          case WordBinopOp::Kind::kBitwiseOr:
            return VisitBinop(selector, node, is64 ? kX64Or : kX64Or32, cont);
          default:
            break;
        }
      }
    } else if (const ShiftOp* shift = op.TryCast<ShiftOp>()) {
      if (selector->IsOnlyUserOfNodeInSameBlock(user, node)) {
        const bool is64 = shift->rep == WordRepresentation::Word64();
        switch (shift->kind) {
          case ShiftOp::Kind::kShiftLeft:
            if (TryVisitWordShift(selector, node, is64 ? 64 : 32,
                                  is64 ? kX64Shl : kX64Shl32, cont)) {
              return;
            }
            break;
          case ShiftOp::Kind::kShiftRightLogical:
            if (TryVisitWordShift(selector, node, is64 ? 64 : 32,
                                  is64 ? kX64Shr : kX64Shr32, cont)) {
              return;
            }
            break;
          default:
            break;
        }
      }
    }
  }

  int effect_level = selector->GetEffectLevel(node, cont);
  if (const auto load = op.TryCast<turboshaft::LoadOp>()) {
    if (load->loaded_rep == turboshaft::MemoryRepresentation::Int8() ||
        load->loaded_rep == turboshaft::MemoryRepresentation::Uint8()) {
      if (opcode == kX64Cmp32) {
        opcode = kX64Cmp8;
      } else if (opcode == kX64Test32) {
        opcode = kX64Test8;
      }
    } else if (load->loaded_rep == turboshaft::MemoryRepresentation::Int16() ||
