template <>
void InstructionSelectorT<TurboshaftAdapter>::VisitNode(
    turboshaft::OpIndex node) {
  using namespace turboshaft;  // NOLINT(build/namespaces)
  tick_counter_->TickAndMaybeEnterSafepoint();
  const turboshaft::Operation& op = this->Get(node);
  using Opcode = turboshaft::Opcode;
  using Rep = turboshaft::RegisterRepresentation;
  switch (op.opcode) {
    case Opcode::kBranch:
    case Opcode::kGoto:
    case Opcode::kReturn:
    case Opcode::kTailCall:
    case Opcode::kUnreachable:
    case Opcode::kDeoptimize:
    case Opcode::kSwitch:
    case Opcode::kCheckException:
      // Those are already handled in VisitControl.
      DCHECK(op.IsBlockTerminator());
      break;
    case Opcode::kParameter: {
      // Parameters should always be scheduled to the first block.
      DCHECK_EQ(this->rpo_number(this->block(schedule(), node)).ToInt(), 0);
      MachineType type = linkage()->GetParameterType(
          op.Cast<turboshaft::ParameterOp>().parameter_index);
      MarkAsRepresentation(type.representation(), node);
      return VisitParameter(node);
    }
    case Opcode::kChange: {
      const turboshaft::ChangeOp& change = op.Cast<turboshaft::ChangeOp>();
      MarkAsRepresentation(change.to.machine_representation(), node);
      switch (change.kind) {
        case ChangeOp::Kind::kFloatConversion:
          if (change.from == Rep::Float64()) {
            DCHECK_EQ(change.to, Rep::Float32());
            return VisitTruncateFloat64ToFloat32(node);
          } else {
            DCHECK_EQ(change.from, Rep::Float32());
            DCHECK_EQ(change.to, Rep::Float64());
            return VisitChangeFloat32ToFloat64(node);
          }
        case ChangeOp::Kind::kSignedFloatTruncateOverflowToMin:
        case ChangeOp::Kind::kUnsignedFloatTruncateOverflowToMin: {
          using A = ChangeOp::Assumption;
          bool is_signed =
              change.kind == ChangeOp::Kind::kSignedFloatTruncateOverflowToMin;
          switch (multi(change.from, change.to, is_signed, change.assumption)) {
            case multi(Rep::Float32(), Rep::Word32(), true, A::kNoOverflow):
            case multi(Rep::Float32(), Rep::Word32(), true, A::kNoAssumption):
              return VisitTruncateFloat32ToInt32(node);
            case multi(Rep::Float32(), Rep::Word32(), false, A::kNoOverflow):
            case multi(Rep::Float32(), Rep::Word32(), false, A::kNoAssumption):
              return VisitTruncateFloat32ToUint32(node);
            case multi(Rep::Float64(), Rep::Word32(), true, A::kReversible):
              return VisitChangeFloat64ToInt32(node);
            case multi(Rep::Float64(), Rep::Word32(), false, A::kReversible):
              return VisitChangeFloat64ToUint32(node);
            case multi(Rep::Float64(), Rep::Word32(), true, A::kNoOverflow):
              return VisitRoundFloat64ToInt32(node);
            case multi(Rep::Float64(), Rep::Word32(), false, A::kNoAssumption):
            case multi(Rep::Float64(), Rep::Word32(), false, A::kNoOverflow):
              return VisitTruncateFloat64ToUint32(node);
            case multi(Rep::Float64(), Rep::Word64(), true, A::kReversible):
              return VisitChangeFloat64ToInt64(node);
            case multi(Rep::Float64(), Rep::Word64(), false, A::kReversible):
              return VisitChangeFloat64ToUint64(node);
            case multi(Rep::Float64(), Rep::Word64(), true, A::kNoOverflow):
            case multi(Rep::Float64(), Rep::Word64(), true, A::kNoAssumption):
              return VisitTruncateFloat64ToInt64(node);
            default:
              // Invalid combination.
              UNREACHABLE();
          }

          UNREACHABLE();
        }
        case ChangeOp::Kind::kJSFloatTruncate:
          DCHECK_EQ(change.from, Rep::Float64());
          DCHECK_EQ(change.to, Rep::Word32());
          return VisitTruncateFloat64ToWord32(node);
        case ChangeOp::Kind::kSignedToFloat:
          if (change.from == Rep::Word32()) {
            if (change.to == Rep::Float32()) {
              return VisitRoundInt32ToFloat32(node);
            } else {
              DCHECK_EQ(change.to, Rep::Float64());
              DCHECK_EQ(change.assumption, ChangeOp::Assumption::kNoAssumption);
              return VisitChangeInt32ToFloat64(node);
            }
          } else {
            DCHECK_EQ(change.from, Rep::Word64());
            if (change.to == Rep::Float32()) {
              return VisitRoundInt64ToFloat32(node);
            } else {
              DCHECK_EQ(change.to, Rep::Float64());
              if (change.assumption == ChangeOp::Assumption::kReversible) {
                return VisitChangeInt64ToFloat64(node);
              } else {
                return VisitRoundInt64ToFloat64(node);
              }
            }
          }
          UNREACHABLE();
        case ChangeOp::Kind::kUnsignedToFloat:
          switch (multi(change.from, change.to)) {
            case multi(Rep::Word32(), Rep::Float32()):
              return VisitRoundUint32ToFloat32(node);
            case multi(Rep::Word32(), Rep::Float64()):
              return VisitChangeUint32ToFloat64(node);
            case multi(Rep::Word64(), Rep::Float32()):
              return VisitRoundUint64ToFloat32(node);
            case multi(Rep::Word64(), Rep::Float64()):
              return VisitRoundUint64ToFloat64(node);
            default:
              UNREACHABLE();
          }
        case ChangeOp::Kind::kExtractHighHalf:
          DCHECK_EQ(change.from, Rep::Float64());
          DCHECK_EQ(change.to, Rep::Word32());
          return VisitFloat64ExtractHighWord32(node);
        case ChangeOp::Kind::kExtractLowHalf:
          DCHECK_EQ(change.from, Rep::Float64());
          DCHECK_EQ(change.to, Rep::Word32());
          return VisitFloat64ExtractLowWord32(node);
        case ChangeOp::Kind::kZeroExtend:
          DCHECK_EQ(change.from, Rep::Word32());
          DCHECK_EQ(change.to, Rep::Word64());
          return VisitChangeUint32ToUint64(node);
        case ChangeOp::Kind::kSignExtend:
          DCHECK_EQ(change.from, Rep::Word32());
          DCHECK_EQ(change.to, Rep::Word64());
          return VisitChangeInt32ToInt64(node);
        case ChangeOp::Kind::kTruncate:
          DCHECK_EQ(change.from, Rep::Word64());
          DCHECK_EQ(change.to, Rep::Word32());
          MarkAsWord32(node);
          return VisitTruncateInt64ToInt32(node);
        case ChangeOp::Kind::kBitcast:
          switch (multi(change.from, change.to)) {
            case multi(Rep::Word32(), Rep::Word64()):
              return VisitBitcastWord32ToWord64(node);
            case multi(Rep::Word32(), Rep::Float32()):
              return VisitBitcastInt32ToFloat32(node);
            case multi(Rep::Word64(), Rep::Float64()):
              return VisitBitcastInt64ToFloat64(node);
            case multi(Rep::Float32(), Rep::Word32()):
              return VisitBitcastFloat32ToInt32(node);
            case multi(Rep::Float64(), Rep::Word64()):
              return VisitBitcastFloat64ToInt64(node);
            default:
              UNREACHABLE();
          }
      }
      UNREACHABLE();
    }
    case Opcode::kTryChange: {
      const TryChangeOp& try_change = op.Cast<TryChangeOp>();
      MarkAsRepresentation(try_change.to.machine_representation(), node);
      DCHECK(try_change.kind ==
                 TryChangeOp::Kind::kSignedFloatTruncateOverflowUndefined ||
             try_change.kind ==
                 TryChangeOp::Kind::kUnsignedFloatTruncateOverflowUndefined);
      const bool is_signed =
          try_change.kind ==
          TryChangeOp::Kind::kSignedFloatTruncateOverflowUndefined;
      switch (multi(try_change.from, try_change.to, is_signed)) {
        case multi(Rep::Float64(), Rep::Word64(), true):
          return VisitTryTruncateFloat64ToInt64(node);
        case multi(Rep::Float64(), Rep::Word64(), false):
          return VisitTryTruncateFloat64ToUint64(node);
        case multi(Rep::Float64(), Rep::Word32(), true):
          return VisitTryTruncateFloat64ToInt32(node);
        case multi(Rep::Float64(), Rep::Word32(), false):
          return VisitTryTruncateFloat64ToUint32(node);
        case multi(Rep::Float32(), Rep::Word64(), true):
          return VisitTryTruncateFloat32ToInt64(node);
        case multi(Rep::Float32(), Rep::Word64(), false):
          return VisitTryTruncateFloat32ToUint64(node);
        default:
          UNREACHABLE();
      }
      UNREACHABLE();
    }
    case Opcode::kConstant: {
      const ConstantOp& constant = op.Cast<ConstantOp>();
      switch (constant.kind) {
        case ConstantOp::Kind::kWord32:
        case ConstantOp::Kind::kWord64:
        case ConstantOp::Kind::kTaggedIndex:
        case ConstantOp::Kind::kExternal:
          break;
        case ConstantOp::Kind::kFloat32:
          MarkAsFloat32(node);
          break;
        case ConstantOp::Kind::kFloat64:
          MarkAsFloat64(node);
          break;
        case ConstantOp::Kind::kHeapObject:
          MarkAsTagged(node);
          break;
        case ConstantOp::Kind::kCompressedHeapObject:
          MarkAsCompressed(node);
          break;
        case ConstantOp::Kind::kNumber:
          if (!IsSmiDouble(constant.number())) MarkAsTagged(node);
          break;
        case ConstantOp::Kind::kRelocatableWasmCall:
        case ConstantOp::Kind::kRelocatableWasmStubCall:
          break;
      }
      VisitConstant(node);
      break;
    }
    case Opcode::kWordUnary: {
      const WordUnaryOp& unop = op.Cast<WordUnaryOp>();
      if (unop.rep == WordRepresentation::Word32()) {
        MarkAsWord32(node);
        switch (unop.kind) {
          case WordUnaryOp::Kind::kReverseBytes:
            return VisitWord32ReverseBytes(node);
          case WordUnaryOp::Kind::kCountLeadingZeros:
            return VisitWord32Clz(node);
          case WordUnaryOp::Kind::kCountTrailingZeros:
            return VisitWord32Ctz(node);
          case WordUnaryOp::Kind::kPopCount:
            return VisitWord32Popcnt(node);
          case WordUnaryOp::Kind::kSignExtend8:
            return VisitSignExtendWord8ToInt32(node);
          case WordUnaryOp::Kind::kSignExtend16:
            return VisitSignExtendWord16ToInt32(node);
        }
      } else {
        DCHECK_EQ(unop.rep, WordRepresentation::Word64());
        MarkAsWord64(node);
        switch (unop.kind) {
          case WordUnaryOp::Kind::kReverseBytes:
            return VisitWord64ReverseBytes(node);
          case WordUnaryOp::Kind::kCountLeadingZeros:
            return VisitWord64Clz(node);
          case WordUnaryOp::Kind::kCountTrailingZeros:
            return VisitWord64Ctz(node);
          case WordUnaryOp::Kind::kPopCount:
            return VisitWord64Popcnt(node);
          case WordUnaryOp::Kind::kSignExtend8:
            return VisitSignExtendWord8ToInt64(node);
          case WordUnaryOp::Kind::kSignExtend16:
            return VisitSignExtendWord16ToInt64(node);
        }
      }
      UNREACHABLE();
    }
    case Opcode::kWordBinop: {
      const WordBinopOp& binop = op.Cast<WordBinopOp>();
      if (binop.rep == WordRepresentation::Word32()) {
        MarkAsWord32(node);
        switch (binop.kind) {
          case WordBinopOp::Kind::kAdd:
            return VisitInt32Add(node);
          case WordBinopOp::Kind::kMul:
            return VisitInt32Mul(node);
          case WordBinopOp::Kind::kSignedMulOverflownBits:
            return VisitInt32MulHigh(node);
          case WordBinopOp::Kind::kUnsignedMulOverflownBits:
            return VisitUint32MulHigh(node);
          case WordBinopOp::Kind::kBitwiseAnd:
            return VisitWord32And(node);
          case WordBinopOp::Kind::kBitwiseOr:
            return VisitWord32Or(node);
          case WordBinopOp::Kind::kBitwiseXor:
            return VisitWord32Xor(node);
          case WordBinopOp::Kind::kSub:
            return VisitInt32Sub(node);
          case WordBinopOp::Kind::kSignedDiv:
            return VisitInt32Div(node);
          case WordBinopOp::Kind::kUnsignedDiv:
            return VisitUint32Div(node);
          case WordBinopOp::Kind::kSignedMod:
            return VisitInt32Mod(node);
          case WordBinopOp::Kind::kUnsignedMod:
            return VisitUint32Mod(node);
        }
      } else {
        DCHECK_EQ(binop.rep, WordRepresentation::Word64());
        MarkAsWord64(node);
        switch (binop.kind) {
          case WordBinopOp::Kind::kAdd:
            return VisitInt64Add(node);
          case WordBinopOp::Kind::kMul:
            return VisitInt64Mul(node);
          case WordBinopOp::Kind::kSignedMulOverflownBits:
            return VisitInt64MulHigh(node);
          case WordBinopOp::Kind::kUnsignedMulOverflownBits:
            return VisitUint64MulHigh(node);
          case WordBinopOp::Kind::kBitwiseAnd:
            return VisitWord64And(node);
          case WordBinopOp::Kind::kBitwiseOr:
            return VisitWord64Or(node);
          case WordBinopOp::Kind::kBitwiseXor:
            return VisitWord64Xor(node);
          case WordBinopOp::Kind::kSub:
            return VisitInt64Sub(node);
          case WordBinopOp::Kind::kSignedDiv:
            return VisitInt64Div(node);
          case WordBinopOp::Kind::kUnsignedDiv:
            return VisitUint64Div(node);
          case WordBinopOp::Kind::kSignedMod:
            return VisitInt64Mod(node);
          case WordBinopOp::Kind::kUnsignedMod:
            return VisitUint64Mod(node);
        }
      }
      UNREACHABLE();
    }
    case Opcode::kFloatUnary: {
      const auto& unop = op.Cast<FloatUnaryOp>();
      if (unop.rep == Rep::Float32()) {
        MarkAsFloat32(node);
        switch (unop.kind) {
          case FloatUnaryOp::Kind::kAbs:
            return VisitFloat32Abs(node);
          case FloatUnaryOp::Kind::kNegate:
            return VisitFloat32Neg(node);
          case FloatUnaryOp::Kind::kRoundDown:
            return VisitFloat32RoundDown(node);
          case FloatUnaryOp::Kind::kRoundUp:
            return VisitFloat32RoundUp(node);
          case FloatUnaryOp::Kind::kRoundToZero:
            return VisitFloat32RoundTruncate(node);
          case FloatUnaryOp::Kind::kRoundTiesEven:
            return VisitFloat32RoundTiesEven(node);
          case FloatUnaryOp::Kind::kSqrt:
            return VisitFloat32Sqrt(node);
          // Those operations are only supported on 64 bit.
          case FloatUnaryOp::Kind::kSilenceNaN:
          case FloatUnaryOp::Kind::kLog:
          case FloatUnaryOp::Kind::kLog2:
          case FloatUnaryOp::Kind::kLog10:
          case FloatUnaryOp::Kind::kLog1p:
          case FloatUnaryOp::Kind::kCbrt:
          case FloatUnaryOp::Kind::kExp:
          case FloatUnaryOp::Kind::kExpm1:
          case FloatUnaryOp::Kind::kSin:
          case FloatUnaryOp::Kind::kCos:
          case FloatUnaryOp::Kind::kSinh:
          case FloatUnaryOp::Kind::kCosh:
          case FloatUnaryOp::Kind::kAcos:
          case FloatUnaryOp::Kind::kAsin:
          case FloatUnaryOp::Kind::kAsinh:
          case FloatUnaryOp::Kind::kAcosh:
          case FloatUnaryOp::Kind::kTan:
          case FloatUnaryOp::Kind::kTanh:
          case FloatUnaryOp::Kind::kAtan:
          case FloatUnaryOp::Kind::kAtanh:
            UNREACHABLE();
        }
      } else {
        DCHECK_EQ(unop.rep, Rep::Float64());
        MarkAsFloat64(node);
        switch (unop.kind) {
          case FloatUnaryOp::Kind::kAbs:
            return VisitFloat64Abs(node);
          case FloatUnaryOp::Kind::kNegate:
            return VisitFloat64Neg(node);
          case FloatUnaryOp::Kind::kSilenceNaN:
            return VisitFloat64SilenceNaN(node);
          case FloatUnaryOp::Kind::kRoundDown:
            return VisitFloat64RoundDown(node);
          case FloatUnaryOp::Kind::kRoundUp:
            return VisitFloat64RoundUp(node);
          case FloatUnaryOp::Kind::kRoundToZero:
            return VisitFloat64RoundTruncate(node);
          case FloatUnaryOp::Kind::kRoundTiesEven:
            return VisitFloat64RoundTiesEven(node);
          case FloatUnaryOp::Kind::kLog:
            return VisitFloat64Log(node);
          case FloatUnaryOp::Kind::kLog2:
            return VisitFloat64Log2(node);
          case FloatUnaryOp::Kind::kLog10:
            return VisitFloat64Log10(node);
          case FloatUnaryOp::Kind::kLog1p:
            return VisitFloat64Log1p(node);
          case FloatUnaryOp::Kind::kSqrt:
            return VisitFloat64Sqrt(node);
          case FloatUnaryOp::Kind::kCbrt:
            return VisitFloat64Cbrt(node);
          case FloatUnaryOp::Kind::kExp:
            return VisitFloat64Exp(node);
          case FloatUnaryOp::Kind::kExpm1:
            return VisitFloat64Expm1(node);
          case FloatUnaryOp::Kind::kSin:
            return VisitFloat64Sin(node);
          case FloatUnaryOp::Kind::kCos:
            return VisitFloat64Cos(node);
          case FloatUnaryOp::Kind::kSinh:
            return VisitFloat64Sinh(node);
          case FloatUnaryOp::Kind::kCosh:
            return VisitFloat64Cosh(node);
          case FloatUnaryOp::Kind::kAcos:
            return VisitFloat64Acos(node);
          case FloatUnaryOp::Kind::kAsin:
            return VisitFloat64Asin(node);
          case FloatUnaryOp::Kind::kAsinh:
            return VisitFloat64Asinh(node);
          case FloatUnaryOp::Kind::kAcosh:
            return VisitFloat64Acosh(node);
          case FloatUnaryOp::Kind::kTan:
            return VisitFloat64Tan(node);
          case FloatUnaryOp::Kind::kTanh:
            return VisitFloat64Tanh(node);
          case FloatUnaryOp::Kind::kAtan:
            return VisitFloat64Atan(node);
          case FloatUnaryOp::Kind::kAtanh:
            return VisitFloat64Atanh(node);
        }
      }
      UNREACHABLE();
    }
    case Opcode::kFloatBinop: {
      const auto& binop = op.Cast<FloatBinopOp>();
      if (binop.rep == Rep::Float32()) {
        MarkAsFloat32(node);
        switch (binop.kind) {
          case FloatBinopOp::Kind::kAdd:
            return VisitFloat32Add(node);
          case FloatBinopOp::Kind::kSub:
            return VisitFloat32Sub(node);
          case FloatBinopOp::Kind::kMul:
            return VisitFloat32Mul(node);
          case FloatBinopOp::Kind::kDiv:
            return VisitFloat32Div(node);
          case FloatBinopOp::Kind::kMin:
            return VisitFloat32Min(node);
          case FloatBinopOp::Kind::kMax:
            return VisitFloat32Max(node);
          case FloatBinopOp::Kind::kMod:
          case FloatBinopOp::Kind::kPower:
          case FloatBinopOp::Kind::kAtan2:
            UNREACHABLE();
        }
      } else {
        DCHECK_EQ(binop.rep, Rep::Float64());
        MarkAsFloat64(node);
        switch (binop.kind) {
          case FloatBinopOp::Kind::kAdd:
            return VisitFloat64Add(node);
          case FloatBinopOp::Kind::kSub:
            return VisitFloat64Sub(node);
          case FloatBinopOp::Kind::kMul:
            return VisitFloat64Mul(node);
          case FloatBinopOp::Kind::kDiv:
            return VisitFloat64Div(node);
          case FloatBinopOp::Kind::kMod:
            return VisitFloat64Mod(node);
          case FloatBinopOp::Kind::kMin:
            return VisitFloat64Min(node);
          case FloatBinopOp::Kind::kMax:
            return VisitFloat64Max(node);
          case FloatBinopOp::Kind::kPower:
            return VisitFloat64Pow(node);
          case FloatBinopOp::Kind::kAtan2:
            return VisitFloat64Atan2(node);
        }
      }
      UNREACHABLE();
    }
    case Opcode::kOverflowCheckedBinop: {
      const auto& binop = op.Cast<OverflowCheckedBinopOp>();
      if (binop.rep == WordRepresentation::Word32()) {
        MarkAsWord32(node);
        switch (binop.kind) {
          case OverflowCheckedBinopOp::Kind::kSignedAdd:
            return VisitInt32AddWithOverflow(node);
          case OverflowCheckedBinopOp::Kind::kSignedMul:
            return VisitInt32MulWithOverflow(node);
          case OverflowCheckedBinopOp::Kind::kSignedSub:
            return VisitInt32SubWithOverflow(node);
        }
      } else {
        DCHECK_EQ(binop.rep, WordRepresentation::Word64());
        MarkAsWord64(node);
        switch (binop.kind) {
          case OverflowCheckedBinopOp::Kind::kSignedAdd:
            return VisitInt64AddWithOverflow(node);
          case OverflowCheckedBinopOp::Kind::kSignedMul:
            return VisitInt64MulWithOverflow(node);
          case OverflowCheckedBinopOp::Kind::kSignedSub:
            return VisitInt64SubWithOverflow(node);
        }
      }
      UNREACHABLE();
    }
    case Opcode::kShift: {
      const auto& shift = op.Cast<ShiftOp>();
      if (shift.rep == RegisterRepresentation::Word32()) {
        MarkAsWord32(node);
        switch (shift.kind) {
          case ShiftOp::Kind::kShiftRightArithmeticShiftOutZeros:
          case ShiftOp::Kind::kShiftRightArithmetic:
            return VisitWord32Sar(node);
          case ShiftOp::Kind::kShiftRightLogical:
            return VisitWord32Shr(node);
          case ShiftOp::Kind::kShiftLeft:
            return VisitWord32Shl(node);
          case ShiftOp::Kind::kRotateRight:
            return VisitWord32Ror(node);
          case ShiftOp::Kind::kRotateLeft:
            return VisitWord32Rol(node);
        }
      } else {
        DCHECK_EQ(shift.rep, RegisterRepresentation::Word64());
        MarkAsWord64(node);
        switch (shift.kind) {
          case ShiftOp::Kind::kShiftRightArithmeticShiftOutZeros:
          case ShiftOp::Kind::kShiftRightArithmetic:
            return VisitWord64Sar(node);
          case ShiftOp::Kind::kShiftRightLogical:
            return VisitWord64Shr(node);
          case ShiftOp::Kind::kShiftLeft:
            return VisitWord64Shl(node);
          case ShiftOp::Kind::kRotateRight:
            return VisitWord64Ror(node);
          case ShiftOp::Kind::kRotateLeft:
            return VisitWord64Rol(node);
        }
      }
      UNREACHABLE();
    }
    case Opcode::kCall:
      // Process the call at `DidntThrow`, when we know if exceptions are caught
      // or not.
      break;
    case Opcode::kDidntThrow:
      if (current_block_->begin() == node) {
        DCHECK_EQ(current_block_->PredecessorCount(), 1);
        DCHECK(current_block_->LastPredecessor()
                   ->LastOperation(*this->turboshaft_graph())
                   .Is<CheckExceptionOp>());
        // In this case, the Call has been generated at the `CheckException`
        // already.
      } else {
        VisitCall(op.Cast<DidntThrowOp>().throwing_operation());
      }
      EmitIdentity(node);
      break;
    case Opcode::kFrameConstant: {
      const auto& constant = op.Cast<turboshaft::FrameConstantOp>();
      using Kind = turboshaft::FrameConstantOp::Kind;
      OperandGenerator g(this);
      switch (constant.kind) {
        case Kind::kStackCheckOffset:
          Emit(kArchStackCheckOffset, g.DefineAsRegister(node));
          break;
        case Kind::kFramePointer:
          Emit(kArchFramePointer, g.DefineAsRegister(node));
          break;
        case Kind::kParentFramePointer:
          Emit(kArchParentFramePointer, g.DefineAsRegister(node));
          break;
      }
      break;
    }
    case Opcode::kStackPointerGreaterThan:
      return VisitStackPointerGreaterThan(node);
    case Opcode::kComparison: {
      const ComparisonOp& comparison = op.Cast<ComparisonOp>();
      using Kind = ComparisonOp::Kind;
      switch (multi(comparison.kind, comparison.rep)) {
        case multi(Kind::kEqual, Rep::Word32()):
          return VisitWord32Equal(node);
        case multi(Kind::kEqual, Rep::Word64()):
          return VisitWord64Equal(node);
        case multi(Kind::kEqual, Rep::Float32()):
          return VisitFloat32Equal(node);
        case multi(Kind::kEqual, Rep::Float64()):
          return VisitFloat64Equal(node);
        case multi(Kind::kEqual, Rep::Tagged()):
          if constexpr (Is64() && !COMPRESS_POINTERS_BOOL) {
            return VisitWord64Equal(node);
          }
          return VisitWord32Equal(node);
        case multi(Kind::kSignedLessThan, Rep::Word32()):
          return VisitInt32LessThan(node);
        case multi(Kind::kSignedLessThan, Rep::Word64()):
          return VisitInt64LessThan(node);
        case multi(Kind::kSignedLessThan, Rep::Float32()):
          return VisitFloat32LessThan(node);
        case multi(Kind::kSignedLessThan, Rep::Float64()):
          return VisitFloat64LessThan(node);
        case multi(Kind::kSignedLessThanOrEqual, Rep::Word32()):
          return VisitInt32LessThanOrEqual(node);
        case multi(Kind::kSignedLessThanOrEqual, Rep::Word64()):
          return VisitInt64LessThanOrEqual(node);
        case multi(Kind::kSignedLessThanOrEqual, Rep::Float32()):
          return VisitFloat32LessThanOrEqual(node);
        case multi(Kind::kSignedLessThanOrEqual, Rep::Float64()):
          return VisitFloat64LessThanOrEqual(node);
        case multi(Kind::kUnsignedLessThan, Rep::Word32()):
          return VisitUint32LessThan(node);
        case multi(Kind::kUnsignedLessThan, Rep::Word64()):
          return VisitUint64LessThan(node);
        case multi(Kind::kUnsignedLessThanOrEqual, Rep::Word32()):
          return VisitUint32LessThanOrEqual(node);
        case multi(Kind::kUnsignedLessThanOrEqual, Rep::Word64()):
          return VisitUint64LessThanOrEqual(node);
        default:
          UNREACHABLE();
      }
      UNREACHABLE();
    }
    case Opcode::kLoad: {
      const LoadOp& load = op.Cast<LoadOp>();
      MachineType loaded_type = load.machine_type();
      MarkAsRepresentation(loaded_type.representation(), node);
      if (load.kind.maybe_unaligned) {
        DCHECK(!load.kind.with_trap_handler);
        if (loaded_type.representation() == MachineRepresentation::kWord8 ||
            InstructionSelector::AlignmentRequirements()
                .IsUnalignedLoadSupported(loaded_type.representation())) {
          return VisitLoad(node);
        } else {
          return VisitUnalignedLoad(node);
        }
      } else if (load.kind.is_atomic) {
        if (load.result_rep == Rep::Word32()) {
          return VisitWord32AtomicLoad(node);
        } else {
          DCHECK_EQ(load.result_rep, Rep::Word64());
          return VisitWord64AtomicLoad(node);
        }
      } else if (load.kind.with_trap_handler) {
        DCHECK(!load.kind.maybe_unaligned);
        return VisitProtectedLoad(node);
      } else {
        return VisitLoad(node);
      }
      UNREACHABLE();
    }
    case Opcode::kStore: {
      const StoreOp& store = op.Cast<StoreOp>();
      MachineRepresentation rep =
          store.stored_rep.ToMachineType().representation();
      if (store.kind.maybe_unaligned) {
        DCHECK(!store.kind.with_trap_handler);
        DCHECK_EQ(store.write_barrier, WriteBarrierKind::kNoWriteBarrier);
        if (rep == MachineRepresentation::kWord8 ||
            InstructionSelector::AlignmentRequirements()
                .IsUnalignedStoreSupported(rep)) {
          return VisitStore(node);
        } else {
          return VisitUnalignedStore(node);
        }
      } else if (store.kind.is_atomic) {
        if (store.stored_rep == MemoryRepresentation::Int64() ||
            store.stored_rep == MemoryRepresentation::Uint64()) {
          return VisitWord64AtomicStore(node);
        } else {
          return VisitWord32AtomicStore(node);
        }
      } else if (store.kind.with_trap_handler) {
        DCHECK(!store.kind.maybe_unaligned);
        return VisitProtectedStore(node);
      } else {
        return VisitStore(node);
      }
      UNREACHABLE();
    }
    case Opcode::kTaggedBitcast: {
      const TaggedBitcastOp& cast = op.Cast<TaggedBitcastOp>();
      switch (multi(cast.from, cast.to)) {
        case multi(Rep::Tagged(), Rep::Word32()):
          MarkAsWord32(node);
          if constexpr (Is64()) {
            DCHECK_EQ(cast.kind, TaggedBitcastOp::Kind::kSmi);
            DCHECK(SmiValuesAre31Bits());
            // TODO(dmercadier): using EmitIdentity here is not ideal, because
            // users of {node} will then use its input, which may not have the
            // Word32 representation. This might in turn lead to the register
            // allocator wrongly tracking Tagged values that are in fact just
            // Smis. However, using VisitBitcastSmiToWord hurts performance
            // because it inserts a gap move which cannot always be eliminated
            // because the operands may have different sizes (and the move is
            // then truncating or extending). As a temporary work-around until
            // the register allocator is fixed, we use VisitBitcastSmiToWord in
            // DEBUG mode to quiet the register allocator verifier.
#ifdef DEBUG
            return VisitBitcastSmiToWord(node);
#else
            return EmitIdentity(node);
#endif
          } else {
            return VisitBitcastTaggedToWord(node);
          }
        case multi(Rep::Tagged(), Rep::Word64()):
          MarkAsWord64(node);
          return VisitBitcastTaggedToWord(node);
        case multi(Rep::Word32(), Rep::Tagged()):
        case multi(Rep::Word64(), Rep::Tagged()):
          if (cast.kind == TaggedBitcastOp::Kind::kSmi) {
            MarkAsRepresentation(MachineRepresentation::kTaggedSigned, node);
            return EmitIdentity(node);
          } else {
            MarkAsTagged(node);
            return VisitBitcastWordToTagged(node);
          }
        case multi(Rep::Compressed(), Rep::Word32()):
          MarkAsWord32(node);
          if (cast.kind == TaggedBitcastOp::Kind::kSmi) {
            return VisitBitcastSmiToWord(node);
          } else {
            return VisitBitcastTaggedToWord(node);
          }
        default:
          UNIMPLEMENTED();
      }
    }
    case Opcode::kPhi:
      MarkAsRepresentation(op.Cast<PhiOp>().rep, node);
      return VisitPhi(node);
    case Opcode::kProjection:
      return VisitProjection(node);
    case Opcode::kDeoptimizeIf:
      if (Get(node).Cast<DeoptimizeIfOp>().negated) {
        return VisitDeoptimizeUnless(node);
      }
      return VisitDeoptimizeIf(node);
#if V8_ENABLE_WEBASSEMBLY
    case Opcode::kTrapIf: {
      const TrapIfOp& trap_if = op.Cast<TrapIfOp>();
      if (trap_if.negated) {
        return VisitTrapUnless(node, trap_if.trap_id);
      }
      return VisitTrapIf(node, trap_if.trap_id);
    }
#endif  // V8_ENABLE_WEBASSEMBLY
    case Opcode::kCatchBlockBegin:
      MarkAsTagged(node);
      return VisitIfException(node);
    case Opcode::kRetain:
      return VisitRetain(node);
    case Opcode::kOsrValue:
      MarkAsTagged(node);
      return VisitOsrValue(node);
    case Opcode::kStackSlot:
      return VisitStackSlot(node);
    case Opcode::kFrameState:
      // FrameState is covered as part of calls.
      UNREACHABLE();
    case Opcode::kLoadRootRegister:
      return VisitLoadRootRegister(node);
    case Opcode::kAssumeMap:
      // AssumeMap is used as a hint for optimization phases but does not
      // produce any code.
      return;
    case Opcode::kDebugBreak:
      return VisitDebugBreak(node);
    case Opcode::kSelect: {
      const SelectOp& select = op.Cast<SelectOp>();
      // If there is a Select, then it should only be one that is supported by
      // the machine, and it should be meant to be implementation with cmove.
      DCHECK_EQ(select.implem, SelectOp::Implementation::kCMove);
      MarkAsRepresentation(select.rep, node);
      return VisitSelect(node);
    }
    case Opcode::kWord32PairBinop: {
      const Word32PairBinopOp& binop = op.Cast<Word32PairBinopOp>();
      MarkAsWord32(node);
      MarkPairProjectionsAsWord32(node);
      switch (binop.kind) {
        case Word32PairBinopOp::Kind::kAdd:
          return VisitInt32PairAdd(node);
        case Word32PairBinopOp::Kind::kSub:
          return VisitInt32PairSub(node);
        case Word32PairBinopOp::Kind::kMul:
          return VisitInt32PairMul(node);
        case Word32PairBinopOp::Kind::kShiftLeft:
          return VisitWord32PairShl(node);
        case Word32PairBinopOp::Kind::kShiftRightLogical:
          return VisitWord32PairShr(node);
        case Word32PairBinopOp::Kind::kShiftRightArithmetic:
          return VisitWord32PairSar(node);
      }
      UNREACHABLE();
    }
    case Opcode::kAtomicWord32Pair: {
      const AtomicWord32PairOp& atomic_op = op.Cast<AtomicWord32PairOp>();
      if (atomic_op.kind != AtomicWord32PairOp::Kind::kStore) {
        MarkAsWord32(node);
        MarkPairProjectionsAsWord32(node);
      }
      switch (atomic_op.kind) {
        case AtomicWord32PairOp::Kind::kAdd:
          return VisitWord32AtomicPairAdd(node);
        case AtomicWord32PairOp::Kind::kAnd:
          return VisitWord32AtomicPairAnd(node);
        case AtomicWord32PairOp::Kind::kCompareExchange:
          return VisitWord32AtomicPairCompareExchange(node);
        case AtomicWord32PairOp::Kind::kExchange:
          return VisitWord32AtomicPairExchange(node);
        case AtomicWord32PairOp::Kind::kLoad:
          return VisitWord32AtomicPairLoad(node);
        case AtomicWord32PairOp::Kind::kOr:
          return VisitWord32AtomicPairOr(node);
        case AtomicWord32PairOp::Kind::kSub:
          return VisitWord32AtomicPairSub(node);
        case AtomicWord32PairOp::Kind::kXor:
          return VisitWord32AtomicPairXor(node);
        case AtomicWord32PairOp::Kind::kStore:
          return VisitWord32AtomicPairStore(node);
      }
    }
    case Opcode::kBitcastWord32PairToFloat64:
      return MarkAsFloat64(node), VisitBitcastWord32PairToFloat64(node);
    case Opcode::kAtomicRMW: {
      const AtomicRMWOp& atomic_op = op.Cast<AtomicRMWOp>();
      MarkAsRepresentation(atomic_op.input_rep.ToRegisterRepresentation(),
                           node);
      if (atomic_op.result_rep == Rep::Word32()) {
        switch (atomic_op.bin_op) {
          case AtomicRMWOp::BinOp::kAdd:
            return VisitWord32AtomicAdd(node);
          case AtomicRMWOp::BinOp::kSub:
            return VisitWord32AtomicSub(node);
          case AtomicRMWOp::BinOp::kAnd:
            return VisitWord32AtomicAnd(node);
          case AtomicRMWOp::BinOp::kOr:
            return VisitWord32AtomicOr(node);
          case AtomicRMWOp::BinOp::kXor:
            return VisitWord32AtomicXor(node);
          case AtomicRMWOp::BinOp::kExchange:
            return VisitWord32AtomicExchange(node);
          case AtomicRMWOp::BinOp::kCompareExchange:
            return VisitWord32AtomicCompareExchange(node);
        }
      } else {
        DCHECK_EQ(atomic_op.result_rep, Rep::Word64());
        switch (atomic_op.bin_op) {
          case AtomicRMWOp::BinOp::kAdd:
            return VisitWord64AtomicAdd(node);
          case AtomicRMWOp::BinOp::kSub:
            return VisitWord64AtomicSub(node);
          case AtomicRMWOp::BinOp::kAnd:
            return VisitWord64AtomicAnd(node);
          case AtomicRMWOp::BinOp::kOr:
            return VisitWord64AtomicOr(node);
          case AtomicRMWOp::BinOp::kXor:
            return VisitWord64AtomicXor(node);
          case AtomicRMWOp::BinOp::kExchange:
            return VisitWord64AtomicExchange(node);
          case AtomicRMWOp::BinOp::kCompareExchange:
            return VisitWord64AtomicCompareExchange(node);
        }
      }
      UNREACHABLE();
    }
    case Opcode::kMemoryBarrier:
      return VisitMemoryBarrier(node);

    case Opcode::kComment:
      return VisitComment(node);

#ifdef V8_ENABLE_WEBASSEMBLY
    case Opcode::kSimd128Constant: {
      const Simd128ConstantOp& constant = op.Cast<Simd128ConstantOp>();
      MarkAsSimd128(node);
      if (constant.IsZero()) return VisitS128Zero(node);
      return VisitS128Const(node);
    }
    case Opcode::kSimd128Unary: {
      const Simd128UnaryOp& unary = op.Cast<Simd128UnaryOp>();
      MarkAsSimd128(node);
      switch (unary.kind) {
#define VISIT_SIMD_UNARY(kind)        \
  case Simd128UnaryOp::Kind::k##kind: \
    return Visit##kind(node);
        FOREACH_SIMD_128_UNARY_OPCODE(VISIT_SIMD_UNARY)
#undef VISIT_SIMD_UNARY
      }
    }
    case Opcode::kSimd128Binop: {
      const Simd128BinopOp& binop = op.Cast<Simd128BinopOp>();
      MarkAsSimd128(node);
      switch (binop.kind) {
#define VISIT_SIMD_BINOP(kind)        \
  case Simd128BinopOp::Kind::k##kind: \
    return Visit##kind(node);
        FOREACH_SIMD_128_BINARY_OPCODE(VISIT_SIMD_BINOP)
#undef VISIT_SIMD_BINOP
      }
    }
    case Opcode::kSimd128Shift: {
      const Simd128ShiftOp& shift = op.Cast<Simd128ShiftOp>();
      MarkAsSimd128(node);
      switch (shift.kind) {
#define VISIT_SIMD_SHIFT(kind)        \
  case Simd128ShiftOp::Kind::k##kind: \
    return Visit##kind(node);
        FOREACH_SIMD_128_SHIFT_OPCODE(VISIT_SIMD_SHIFT)
#undef VISIT_SIMD_SHIFT
      }
    }
    case Opcode::kSimd128Test: {
      const Simd128TestOp& test = op.Cast<Simd128TestOp>();
      MarkAsWord32(node);
      switch (test.kind) {
#define VISIT_SIMD_TEST(kind)        \
  case Simd128TestOp::Kind::k##kind: \
    return Visit##kind(node);
        FOREACH_SIMD_128_TEST_OPCODE(VISIT_SIMD_TEST)
#undef VISIT_SIMD_TEST
      }
    }
    case Opcode::kSimd128Splat: {
      const Simd128SplatOp& splat = op.Cast<Simd128SplatOp>();
      MarkAsSimd128(node);
      switch (splat.kind) {
#define VISIT_SIMD_SPLAT(kind)        \
  case Simd128SplatOp::Kind::k##kind: \
    return Visit##kind##Splat(node);
        FOREACH_SIMD_128_SPLAT_OPCODE(VISIT_SIMD_SPLAT)
#undef VISIT_SIMD_SPLAT
      }
    }
    case Opcode::kSimd128Shuffle:
      MarkAsSimd128(node);
      return VisitI8x16Shuffle(node);
    case Opcode::kSimd128ReplaceLane: {
      const Simd128ReplaceLaneOp& replace = op.Cast<Simd128ReplaceLaneOp>();
      MarkAsSimd128(node);
      switch (replace.kind) {
        case Simd128ReplaceLaneOp::Kind::kI8x16:
          return VisitI8x16ReplaceLane(node);
        case Simd128ReplaceLaneOp::Kind::kI16x8:
          return VisitI16x8ReplaceLane(node);
        case Simd128ReplaceLaneOp::Kind::kI32x4:
          return VisitI32x4ReplaceLane(node);
        case Simd128ReplaceLaneOp::Kind::kI64x2:
          return VisitI64x2ReplaceLane(node);
        case Simd128ReplaceLaneOp::Kind::kF32x4:
          return VisitF32x4ReplaceLane(node);
        case Simd128ReplaceLaneOp::Kind::kF64x2:
          return VisitF64x2ReplaceLane(node);
      }
    }
    case Opcode::kSimd128ExtractLane: {
      const Simd128ExtractLaneOp& extract = op.Cast<Simd128ExtractLaneOp>();
      switch (extract.kind) {
        case Simd128ExtractLaneOp::Kind::kI8x16S:
          MarkAsWord32(node);
          return VisitI8x16ExtractLaneS(node);
        case Simd128ExtractLaneOp::Kind::kI8x16U:
          MarkAsWord32(node);
          return VisitI8x16ExtractLaneU(node);
        case Simd128ExtractLaneOp::Kind::kI16x8S:
          MarkAsWord32(node);
          return VisitI16x8ExtractLaneS(node);
        case Simd128ExtractLaneOp::Kind::kI16x8U:
          MarkAsWord32(node);
          return VisitI16x8ExtractLaneU(node);
        case Simd128ExtractLaneOp::Kind::kI32x4:
          MarkAsWord32(node);
          return VisitI32x4ExtractLane(node);
        case Simd128ExtractLaneOp::Kind::kI64x2:
          MarkAsWord64(node);
          return VisitI64x2ExtractLane(node);
        case Simd128ExtractLaneOp::Kind::kF32x4:
          MarkAsFloat32(node);
          return VisitF32x4ExtractLane(node);
        case Simd128ExtractLaneOp::Kind::kF64x2:
          MarkAsFloat64(node);
          return VisitF64x2ExtractLane(node);
      }
    }
    case Opcode::kSimd128LoadTransform:
      MarkAsSimd128(node);
      return VisitLoadTransform(node);
    case Opcode::kSimd128LaneMemory: {
      const Simd128LaneMemoryOp& memory = op.Cast<Simd128LaneMemoryOp>();
      MarkAsSimd128(node);
      if (memory.mode == Simd128LaneMemoryOp::Mode::kLoad) {
        return VisitLoadLane(node);
      } else {
        DCHECK_EQ(memory.mode, Simd128LaneMemoryOp::Mode::kStore);
        return VisitStoreLane(node);
      }
    }
    case Opcode::kSimd128Ternary: {
      const Simd128TernaryOp& ternary = op.Cast<Simd128TernaryOp>();
      MarkAsSimd128(node);
      switch (ternary.kind) {
#define VISIT_SIMD_TERNARY(kind)        \
  case Simd128TernaryOp::Kind::k##kind: \
    return Visit##kind(node);
        FOREACH_SIMD_128_TERNARY_OPCODE(VISIT_SIMD_TERNARY)
#undef VISIT_SIMD_TERNARY
      }
    }

    case Opcode::kLoadStackPointer:
      return VisitLoadStackPointer(node);

    case Opcode::kSetStackPointer:
      return VisitSetStackPointer(node);

#endif  // V8_ENABLE_WEBASSEMBLY

#define UNREACHABLE_CASE(op) case Opcode::k##op:
      TURBOSHAFT_JS_OPERATION_LIST(UNREACHABLE_CASE)
      TURBOSHAFT_SIMPLIFIED_OPERATION_LIST(UNREACHABLE_CASE)
      TURBOSHAFT_WASM_OPERATION_LIST(UNREACHABLE_CASE)
      TURBOSHAFT_OTHER_OPERATION_LIST(UNREACHABLE_CASE)
      UNREACHABLE_CASE(PendingLoopPhi)
      UNREACHABLE_CASE(Tuple)
      UNREACHABLE();
#undef UNREACHABLE_CASE
  }
}
