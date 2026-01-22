  void VisitNode(Node* node, Truncation truncation,
                 SimplifiedLowering* lowering) {
    // Unconditionally eliminate unused pure nodes (only relevant if there's
    // a pure operation in between two effectful ones, where the last one
    // is unused).
    // Note: We must not do this for constants, as they are cached and we
    // would thus kill the cached {node} during lowering (i.e. replace all
    // uses with Dead), but at that point some node lowering might have
    // already taken the constant {node} from the cache (while it was in
    // a sane state still) and we would afterwards replace that use with
    // Dead as well.
    if (node->op()->ValueInputCount() > 0 &&
        node->op()->HasProperty(Operator::kPure)) {
      if (truncation.IsUnused()) return VisitUnused(node);
    }
    switch (node->opcode()) {
      //------------------------------------------------------------------
      // Common operators.
      //------------------------------------------------------------------
      case IrOpcode::kStart:
        // We use Start as a terminator for the frame state chain, so even
        // tho Start doesn't really produce a value, we have to say Tagged
        // here, otherwise the input conversion will fail.
        return VisitLeaf(node, MachineRepresentation::kTagged);
      case IrOpcode::kParameter:
        // TODO(titzer): use representation from linkage.
        return VisitUnop(node, UseInfo::None(), MachineRepresentation::kTagged);
      case IrOpcode::kInt32Constant:
        return VisitLeaf(node, MachineRepresentation::kWord32);
      case IrOpcode::kInt64Constant:
        return VisitLeaf(node, MachineRepresentation::kWord64);
      case IrOpcode::kExternalConstant:
        return VisitLeaf(node, MachineType::PointerRepresentation());
      case IrOpcode::kNumberConstant: {
        double const value = OpParameter<double>(node->op());
        int value_as_int;
        if (DoubleToSmiInteger(value, &value_as_int)) {
          VisitLeaf(node, MachineRepresentation::kTaggedSigned);
          if (lower()) {
            intptr_t smi = bit_cast<intptr_t>(Smi::FromInt(value_as_int));
            DeferReplacement(node, lowering->jsgraph()->IntPtrConstant(smi));
          }
          return;
        }
        VisitLeaf(node, MachineRepresentation::kTagged);
        return;
      }
      case IrOpcode::kHeapConstant:
        return VisitLeaf(node, MachineRepresentation::kTaggedPointer);
      case IrOpcode::kPointerConstant: {
        VisitLeaf(node, MachineType::PointerRepresentation());
        if (lower()) {
          intptr_t const value = OpParameter<intptr_t>(node->op());
          DeferReplacement(node, lowering->jsgraph()->IntPtrConstant(value));
        }
        return;
      }

      case IrOpcode::kBranch: {
        DCHECK(TypeOf(node->InputAt(0))->Is(Type::Boolean()));
        ProcessInput(node, 0, UseInfo::Bool());
        EnqueueInput(node, NodeProperties::FirstControlIndex(node));
        return;
      }
      case IrOpcode::kSwitch:
        ProcessInput(node, 0, UseInfo::TruncatingWord32());
        EnqueueInput(node, NodeProperties::FirstControlIndex(node));
        return;
      case IrOpcode::kSelect:
        return VisitSelect(node, truncation, lowering);
      case IrOpcode::kPhi:
        return VisitPhi(node, truncation, lowering);
      case IrOpcode::kCall:
        return VisitCall(node, lowering);

      //------------------------------------------------------------------
      // JavaScript operators.
      //------------------------------------------------------------------
      case IrOpcode::kToBoolean: {
        if (truncation.IsUsedAsBool()) {
          ProcessInput(node, 0, UseInfo::Bool());
          SetOutput(node, MachineRepresentation::kBit);
          if (lower()) DeferReplacement(node, node->InputAt(0));
        } else {
          VisitInputs(node);
          SetOutput(node, MachineRepresentation::kTaggedPointer);
        }
        return;
      }
      case IrOpcode::kJSToNumber:
      case IrOpcode::kJSToNumeric: {
        VisitInputs(node);
        // TODO(bmeurer): Optimize somewhat based on input type?
        if (truncation.IsUsedAsWord32()) {
          SetOutput(node, MachineRepresentation::kWord32);
          if (lower())
            lowering->DoJSToNumberOrNumericTruncatesToWord32(node, this);
        } else if (truncation.IsUsedAsFloat64()) {
          SetOutput(node, MachineRepresentation::kFloat64);
          if (lower())
            lowering->DoJSToNumberOrNumericTruncatesToFloat64(node, this);
        } else {
          SetOutput(node, MachineRepresentation::kTagged);
        }
        return;
      }

      //------------------------------------------------------------------
      // Simplified operators.
      //------------------------------------------------------------------
      case IrOpcode::kBooleanNot: {
        if (lower()) {
          NodeInfo* input_info = GetInfo(node->InputAt(0));
          if (input_info->representation() == MachineRepresentation::kBit) {
            // BooleanNot(x: kRepBit) => Word32Equal(x, #0)
            node->AppendInput(jsgraph_->zone(), jsgraph_->Int32Constant(0));
            NodeProperties::ChangeOp(node, lowering->machine()->Word32Equal());
          } else if (CanBeTaggedPointer(input_info->representation())) {
            // BooleanNot(x: kRepTagged) => WordEqual(x, #false)
            node->AppendInput(jsgraph_->zone(), jsgraph_->FalseConstant());
            NodeProperties::ChangeOp(node, lowering->machine()->WordEqual());
          } else {
            DCHECK(TypeOf(node->InputAt(0))->IsNone());
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(0));
          }
        } else {
          // No input representation requirement; adapt during lowering.
          ProcessInput(node, 0, UseInfo::AnyTruncatingToBool());
          SetOutput(node, MachineRepresentation::kBit);
        }
        return;
      }
      case IrOpcode::kNumberEqual: {
        Type* const lhs_type = TypeOf(node->InputAt(0));
        Type* const rhs_type = TypeOf(node->InputAt(1));
        // Number comparisons reduce to integer comparisons for integer inputs.
        if ((lhs_type->Is(Type::Unsigned32()) &&
             rhs_type->Is(Type::Unsigned32())) ||
            (lhs_type->Is(Type::Unsigned32OrMinusZeroOrNaN()) &&
             rhs_type->Is(Type::Unsigned32OrMinusZeroOrNaN()) &&
             OneInputCannotBe(node, type_cache_.kZeroish))) {
          // => unsigned Int32Cmp
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     MachineRepresentation::kBit);
          if (lower()) NodeProperties::ChangeOp(node, Uint32Op(node));
          return;
        }
        if ((lhs_type->Is(Type::Signed32()) &&
             rhs_type->Is(Type::Signed32())) ||
            (lhs_type->Is(Type::Signed32OrMinusZeroOrNaN()) &&
             rhs_type->Is(Type::Signed32OrMinusZeroOrNaN()) &&
             OneInputCannotBe(node, type_cache_.kZeroish))) {
          // => signed Int32Cmp
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     MachineRepresentation::kBit);
          if (lower()) NodeProperties::ChangeOp(node, Int32Op(node));
          return;
        }
        // => Float64Cmp
        VisitBinop(node, UseInfo::TruncatingFloat64(),
                   MachineRepresentation::kBit);
        if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kNumberLessThan:
      case IrOpcode::kNumberLessThanOrEqual: {
        // Number comparisons reduce to integer comparisons for integer inputs.
        if (TypeOf(node->InputAt(0))->Is(Type::Unsigned32()) &&
            TypeOf(node->InputAt(1))->Is(Type::Unsigned32())) {
          // => unsigned Int32Cmp
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     MachineRepresentation::kBit);
          if (lower()) NodeProperties::ChangeOp(node, Uint32Op(node));
        } else if (TypeOf(node->InputAt(0))->Is(Type::Signed32()) &&
                   TypeOf(node->InputAt(1))->Is(Type::Signed32())) {
          // => signed Int32Cmp
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     MachineRepresentation::kBit);
          if (lower()) NodeProperties::ChangeOp(node, Int32Op(node));
        } else {
          // => Float64Cmp
          VisitBinop(node, UseInfo::TruncatingFloat64(),
                     MachineRepresentation::kBit);
          if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        }
        return;
      }

      case IrOpcode::kSpeculativeSafeIntegerAdd:
      case IrOpcode::kSpeculativeSafeIntegerSubtract:
        return VisitSpeculativeIntegerAdditiveOp(node, truncation, lowering);

      case IrOpcode::kSpeculativeNumberAdd:
      case IrOpcode::kSpeculativeNumberSubtract:
        return VisitSpeculativeAdditiveOp(node, truncation, lowering);

      case IrOpcode::kSpeculativeNumberLessThan:
      case IrOpcode::kSpeculativeNumberLessThanOrEqual:
      case IrOpcode::kSpeculativeNumberEqual: {
        // Number comparisons reduce to integer comparisons for integer inputs.
        if (TypeOf(node->InputAt(0))->Is(Type::Unsigned32()) &&
            TypeOf(node->InputAt(1))->Is(Type::Unsigned32())) {
          // => unsigned Int32Cmp
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     MachineRepresentation::kBit);
          if (lower()) ChangeToPureOp(node, Uint32Op(node));
          return;
        } else if (TypeOf(node->InputAt(0))->Is(Type::Signed32()) &&
                   TypeOf(node->InputAt(1))->Is(Type::Signed32())) {
          // => signed Int32Cmp
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     MachineRepresentation::kBit);
          if (lower()) ChangeToPureOp(node, Int32Op(node));
          return;
        }
        // Try to use type feedback.
        NumberOperationHint hint = NumberOperationHintOf(node->op());
        switch (hint) {
          case NumberOperationHint::kSigned32:
          case NumberOperationHint::kSignedSmall:
            if (propagate()) {
              VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                         MachineRepresentation::kBit);
            } else if (retype()) {
              SetOutput(node, MachineRepresentation::kBit, Type::Any());
            } else {
              DCHECK(lower());
              Node* lhs = node->InputAt(0);
              Node* rhs = node->InputAt(1);
              if (IsNodeRepresentationTagged(lhs) &&
                  IsNodeRepresentationTagged(rhs)) {
                VisitBinop(
                    node,
                    UseInfo::CheckedSignedSmallAsTaggedSigned(VectorSlotPair()),
                    MachineRepresentation::kBit);
                ChangeToPureOp(
                    node, changer_->TaggedSignedOperatorFor(node->opcode()));

              } else {
                VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                           MachineRepresentation::kBit);
                ChangeToPureOp(node, Int32Op(node));
              }
            }
            return;
          case NumberOperationHint::kSignedSmallInputs:
            // This doesn't make sense for compare operations.
            UNREACHABLE();
          case NumberOperationHint::kNumberOrOddball:
            // Abstract and strict equality don't perform ToNumber conversions
            // on Oddballs, so make sure we don't accidentially sneak in a
            // hint with Oddball feedback here.
            DCHECK_NE(IrOpcode::kSpeculativeNumberEqual, node->opcode());
            V8_FALLTHROUGH;
          case NumberOperationHint::kNumber:
            VisitBinop(node,
                       CheckedUseInfoAsFloat64FromHint(hint, VectorSlotPair()),
                       MachineRepresentation::kBit);
            if (lower()) ChangeToPureOp(node, Float64Op(node));
            return;
        }
        UNREACHABLE();
        return;
      }

      case IrOpcode::kNumberAdd:
      case IrOpcode::kNumberSubtract: {
        if (BothInputsAre(node, type_cache_.kAdditiveSafeIntegerOrMinusZero) &&
            (GetUpperBound(node)->Is(Type::Signed32()) ||
             GetUpperBound(node)->Is(Type::Unsigned32()) ||
             truncation.IsUsedAsWord32())) {
          // => Int32Add/Sub
          VisitWord32TruncatingBinop(node);
          if (lower()) ChangeToPureOp(node, Int32Op(node));
        } else {
          // => Float64Add/Sub
          VisitFloat64Binop(node);
          if (lower()) ChangeToPureOp(node, Float64Op(node));
        }
        return;
      }
      case IrOpcode::kSpeculativeNumberMultiply: {
        if (BothInputsAre(node, Type::Integral32()) &&
            (NodeProperties::GetType(node)->Is(Type::Signed32()) ||
             NodeProperties::GetType(node)->Is(Type::Unsigned32()) ||
             (truncation.IsUsedAsWord32() &&
              NodeProperties::GetType(node)->Is(
                  type_cache_.kSafeIntegerOrMinusZero)))) {
          // Multiply reduces to Int32Mul if the inputs are integers, and
          // (a) the output is either known to be Signed32, or
          // (b) the output is known to be Unsigned32, or
          // (c) the uses are truncating and the result is in the safe
          //     integer range.
          VisitWord32TruncatingBinop(node);
          if (lower()) ChangeToPureOp(node, Int32Op(node));
          return;
        }
        // Try to use type feedback.
        NumberOperationHint hint = NumberOperationHintOf(node->op());
        Type* input0_type = TypeOf(node->InputAt(0));
        Type* input1_type = TypeOf(node->InputAt(1));

        // Handle the case when no int32 checks on inputs are necessary
        // (but an overflow check is needed on the output).
        if (BothInputsAre(node, Type::Signed32())) {
          // If both inputs and feedback are int32, use the overflow op.
          if (hint == NumberOperationHint::kSignedSmall ||
              hint == NumberOperationHint::kSigned32) {
            VisitBinop(node, UseInfo::TruncatingWord32(),
                       MachineRepresentation::kWord32, Type::Signed32());
            if (lower()) {
              LowerToCheckedInt32Mul(node, truncation, input0_type,
                                     input1_type);
            }
            return;
          }
        }

        if (hint == NumberOperationHint::kSignedSmall ||
            hint == NumberOperationHint::kSigned32) {
          VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                     MachineRepresentation::kWord32, Type::Signed32());
          if (lower()) {
            LowerToCheckedInt32Mul(node, truncation, input0_type, input1_type);
          }
          return;
        }

        // Checked float64 x float64 => float64
        VisitBinop(node,
                   UseInfo::CheckedNumberOrOddballAsFloat64(VectorSlotPair()),
                   MachineRepresentation::kFloat64, Type::Number());
        if (lower()) ChangeToPureOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kNumberMultiply: {
        if (BothInputsAre(node, Type::Integral32()) &&
            (NodeProperties::GetType(node)->Is(Type::Signed32()) ||
             NodeProperties::GetType(node)->Is(Type::Unsigned32()) ||
             (truncation.IsUsedAsWord32() &&
              NodeProperties::GetType(node)->Is(
                  type_cache_.kSafeIntegerOrMinusZero)))) {
          // Multiply reduces to Int32Mul if the inputs are integers, and
          // (a) the output is either known to be Signed32, or
          // (b) the output is known to be Unsigned32, or
          // (c) the uses are truncating and the result is in the safe
          //     integer range.
          VisitWord32TruncatingBinop(node);
          if (lower()) ChangeToPureOp(node, Int32Op(node));
          return;
        }
        // Number x Number => Float64Mul
        VisitFloat64Binop(node);
        if (lower()) ChangeToPureOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kSpeculativeNumberDivide: {
        if (BothInputsAreUnsigned32(node) && truncation.IsUsedAsWord32()) {
          // => unsigned Uint32Div
          VisitWord32TruncatingBinop(node);
          if (lower()) DeferReplacement(node, lowering->Uint32Div(node));
          return;
        }
        if (BothInputsAreSigned32(node)) {
          if (NodeProperties::GetType(node)->Is(Type::Signed32())) {
            // => signed Int32Div
            VisitWord32TruncatingBinop(node);
            if (lower()) DeferReplacement(node, lowering->Int32Div(node));
            return;
          }
          if (truncation.IsUsedAsWord32()) {
            // => signed Int32Div
            VisitWord32TruncatingBinop(node);
            if (lower()) DeferReplacement(node, lowering->Int32Div(node));
            return;
          }
        }

        // Try to use type feedback.
        NumberOperationHint hint = NumberOperationHintOf(node->op());

        // Handle the case when no uint32 checks on inputs are necessary
        // (but an overflow check is needed on the output).
        if (BothInputsAreUnsigned32(node)) {
          if (hint == NumberOperationHint::kSignedSmall ||
              hint == NumberOperationHint::kSigned32) {
            VisitBinop(node, UseInfo::TruncatingWord32(),
                       MachineRepresentation::kWord32, Type::Unsigned32());
            if (lower()) ChangeToUint32OverflowOp(node);
            return;
          }
        }

        // Handle the case when no int32 checks on inputs are necessary
        // (but an overflow check is needed on the output).
        if (BothInputsAreSigned32(node)) {
          // If both the inputs the feedback are int32, use the overflow op.
          if (hint == NumberOperationHint::kSignedSmall ||
              hint == NumberOperationHint::kSigned32) {
            VisitBinop(node, UseInfo::TruncatingWord32(),
                       MachineRepresentation::kWord32, Type::Signed32());
            if (lower()) ChangeToInt32OverflowOp(node);
            return;
          }
        }

        if (hint == NumberOperationHint::kSigned32 ||
            hint == NumberOperationHint::kSignedSmall ||
            hint == NumberOperationHint::kSignedSmallInputs) {
          // If the result is truncated, we only need to check the inputs.
          if (truncation.IsUsedAsWord32()) {
            VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                       MachineRepresentation::kWord32);
            if (lower()) DeferReplacement(node, lowering->Int32Div(node));
            return;
          } else if (hint != NumberOperationHint::kSignedSmallInputs) {
            VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                       MachineRepresentation::kWord32, Type::Signed32());
            if (lower()) ChangeToInt32OverflowOp(node);
            return;
          }
        }

        // default case => Float64Div
        VisitBinop(node,
                   UseInfo::CheckedNumberOrOddballAsFloat64(VectorSlotPair()),
                   MachineRepresentation::kFloat64, Type::Number());
        if (lower()) ChangeToPureOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kNumberDivide: {
        if (BothInputsAreUnsigned32(node) && truncation.IsUsedAsWord32()) {
          // => unsigned Uint32Div
          VisitWord32TruncatingBinop(node);
          if (lower()) DeferReplacement(node, lowering->Uint32Div(node));
          return;
        }
        if (BothInputsAreSigned32(node)) {
          if (NodeProperties::GetType(node)->Is(Type::Signed32())) {
            // => signed Int32Div
            VisitWord32TruncatingBinop(node);
            if (lower()) DeferReplacement(node, lowering->Int32Div(node));
            return;
          }
          if (truncation.IsUsedAsWord32()) {
            // => signed Int32Div
            VisitWord32TruncatingBinop(node);
            if (lower()) DeferReplacement(node, lowering->Int32Div(node));
            return;
          }
        }
        // Number x Number => Float64Div
        VisitFloat64Binop(node);
        if (lower()) ChangeToPureOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kSpeculativeNumberModulus:
        return VisitSpeculativeNumberModulus(node, truncation, lowering);
      case IrOpcode::kNumberModulus: {
        if (BothInputsAre(node, Type::Unsigned32OrMinusZeroOrNaN()) &&
            (truncation.IsUsedAsWord32() ||
             NodeProperties::GetType(node)->Is(Type::Unsigned32()))) {
          // => unsigned Uint32Mod
          VisitWord32TruncatingBinop(node);
          if (lower()) DeferReplacement(node, lowering->Uint32Mod(node));
          return;
        }
        if (BothInputsAre(node, Type::Signed32OrMinusZeroOrNaN()) &&
            (truncation.IsUsedAsWord32() ||
             NodeProperties::GetType(node)->Is(Type::Signed32()))) {
          // => signed Int32Mod
          VisitWord32TruncatingBinop(node);
          if (lower()) DeferReplacement(node, lowering->Int32Mod(node));
          return;
        }
        if (TypeOf(node->InputAt(0))->Is(Type::Unsigned32()) &&
            TypeOf(node->InputAt(1))->Is(Type::Unsigned32()) &&
            (truncation.IsUsedAsWord32() ||
             NodeProperties::GetType(node)->Is(Type::Unsigned32()))) {
          // We can only promise Float64 truncation here, as the decision is
          // based on the feedback types of the inputs.
          VisitBinop(node, UseInfo(MachineRepresentation::kWord32,
                                   Truncation::Float64()),
                     MachineRepresentation::kWord32);
          if (lower()) DeferReplacement(node, lowering->Uint32Mod(node));
          return;
        }
        if (TypeOf(node->InputAt(0))->Is(Type::Signed32()) &&
            TypeOf(node->InputAt(1))->Is(Type::Signed32()) &&
            (truncation.IsUsedAsWord32() ||
             NodeProperties::GetType(node)->Is(Type::Signed32()))) {
          // We can only promise Float64 truncation here, as the decision is
          // based on the feedback types of the inputs.
          VisitBinop(node, UseInfo(MachineRepresentation::kWord32,
                                   Truncation::Float64()),
                     MachineRepresentation::kWord32);
          if (lower()) DeferReplacement(node, lowering->Int32Mod(node));
          return;
        }
        // default case => Float64Mod
        VisitFloat64Binop(node);
        if (lower()) ChangeToPureOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kNumberBitwiseOr:
      case IrOpcode::kNumberBitwiseXor:
      case IrOpcode::kNumberBitwiseAnd: {
        VisitWord32TruncatingBinop(node);
        if (lower()) NodeProperties::ChangeOp(node, Int32Op(node));
        return;
      }
      case IrOpcode::kSpeculativeNumberBitwiseOr:
      case IrOpcode::kSpeculativeNumberBitwiseXor:
      case IrOpcode::kSpeculativeNumberBitwiseAnd:
        VisitSpeculativeInt32Binop(node);
        if (lower()) {
          ChangeToPureOp(node, Int32Op(node));
        }
        return;
      case IrOpcode::kNumberShiftLeft: {
        Type* rhs_type = GetUpperBound(node->InputAt(1));
        VisitBinop(node, UseInfo::TruncatingWord32(),
                   UseInfo::TruncatingWord32(), MachineRepresentation::kWord32);
        if (lower()) {
          lowering->DoShift(node, lowering->machine()->Word32Shl(), rhs_type);
        }
        return;
      }
      case IrOpcode::kSpeculativeNumberShiftLeft: {
        if (BothInputsAre(node, Type::NumberOrOddball())) {
          Type* rhs_type = GetUpperBound(node->InputAt(1));
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     UseInfo::TruncatingWord32(),
                     MachineRepresentation::kWord32);
          if (lower()) {
            lowering->DoShift(node, lowering->machine()->Word32Shl(), rhs_type);
          }
          return;
        }
        NumberOperationHint hint = NumberOperationHintOf(node->op());
        Type* rhs_type = GetUpperBound(node->InputAt(1));
        VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                   MachineRepresentation::kWord32, Type::Signed32());
        if (lower()) {
          lowering->DoShift(node, lowering->machine()->Word32Shl(), rhs_type);
        }
        return;
      }
      case IrOpcode::kNumberShiftRight: {
        Type* rhs_type = GetUpperBound(node->InputAt(1));
        VisitBinop(node, UseInfo::TruncatingWord32(),
                   UseInfo::TruncatingWord32(), MachineRepresentation::kWord32);
        if (lower()) {
          lowering->DoShift(node, lowering->machine()->Word32Sar(), rhs_type);
        }
        return;
      }
      case IrOpcode::kSpeculativeNumberShiftRight: {
        if (BothInputsAre(node, Type::NumberOrOddball())) {
          Type* rhs_type = GetUpperBound(node->InputAt(1));
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     UseInfo::TruncatingWord32(),
                     MachineRepresentation::kWord32);
          if (lower()) {
            lowering->DoShift(node, lowering->machine()->Word32Sar(), rhs_type);
          }
          return;
        }
        NumberOperationHint hint = NumberOperationHintOf(node->op());
        Type* rhs_type = GetUpperBound(node->InputAt(1));
        VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                   MachineRepresentation::kWord32, Type::Signed32());
        if (lower()) {
          lowering->DoShift(node, lowering->machine()->Word32Sar(), rhs_type);
        }
        return;
      }
      case IrOpcode::kNumberShiftRightLogical: {
        Type* rhs_type = GetUpperBound(node->InputAt(1));
        VisitBinop(node, UseInfo::TruncatingWord32(),
                   UseInfo::TruncatingWord32(), MachineRepresentation::kWord32);
        if (lower()) {
          lowering->DoShift(node, lowering->machine()->Word32Shr(), rhs_type);
        }
        return;
      }
      case IrOpcode::kSpeculativeNumberShiftRightLogical: {
        NumberOperationHint hint = NumberOperationHintOf(node->op());
        Type* rhs_type = GetUpperBound(node->InputAt(1));
        if (rhs_type->Is(type_cache_.kZeroish) &&
            (hint == NumberOperationHint::kSignedSmall ||
             hint == NumberOperationHint::kSigned32) &&
            !truncation.IsUsedAsWord32()) {
          // The SignedSmall or Signed32 feedback means that the results that we
          // have seen so far were of type Unsigned31.  We speculate that this
          // will continue to hold.  Moreover, since the RHS is 0, the result
          // will just be the (converted) LHS.
          VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                     MachineRepresentation::kWord32, Type::Unsigned31());
          if (lower()) {
            node->RemoveInput(1);
            NodeProperties::ChangeOp(
                node, simplified()->CheckedUint32ToInt32(VectorSlotPair()));
          }
          return;
        }
        if (BothInputsAre(node, Type::NumberOrOddball())) {
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     UseInfo::TruncatingWord32(),
                     MachineRepresentation::kWord32);
          if (lower()) {
            lowering->DoShift(node, lowering->machine()->Word32Shr(), rhs_type);
          }
          return;
        }
        VisitBinop(node, CheckedUseInfoAsWord32FromHint(hint),
                   MachineRepresentation::kWord32, Type::Unsigned32());
        if (lower()) {
          lowering->DoShift(node, lowering->machine()->Word32Shr(), rhs_type);
        }
        return;
      }
      case IrOpcode::kNumberAbs: {
        if (TypeOf(node->InputAt(0))->Is(Type::Unsigned32())) {
          VisitUnop(node, UseInfo::TruncatingWord32(),
                    MachineRepresentation::kWord32);
          if (lower()) DeferReplacement(node, node->InputAt(0));
        } else if (TypeOf(node->InputAt(0))->Is(Type::Signed32())) {
          VisitUnop(node, UseInfo::TruncatingWord32(),
                    MachineRepresentation::kWord32);
          if (lower()) DeferReplacement(node, lowering->Int32Abs(node));
        } else if (TypeOf(node->InputAt(0))
                       ->Is(type_cache_.kPositiveIntegerOrMinusZeroOrNaN)) {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kFloat64);
          if (lower()) DeferReplacement(node, node->InputAt(0));
        } else {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kFloat64);
          if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        }
        return;
      }
      case IrOpcode::kNumberClz32: {
        VisitUnop(node, UseInfo::TruncatingWord32(),
                  MachineRepresentation::kWord32);
        if (lower()) NodeProperties::ChangeOp(node, Uint32Op(node));
        return;
      }
      case IrOpcode::kNumberImul: {
        VisitBinop(node, UseInfo::TruncatingWord32(),
                   UseInfo::TruncatingWord32(), MachineRepresentation::kWord32);
        if (lower()) NodeProperties::ChangeOp(node, Uint32Op(node));
        return;
      }
      case IrOpcode::kNumberFround: {
        VisitUnop(node, UseInfo::TruncatingFloat64(),
                  MachineRepresentation::kFloat32);
        if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kNumberMax: {
        // It is safe to use the feedback types for left and right hand side
        // here, since we can only narrow those types and thus we can only
        // promise a more specific truncation.
        Type* const lhs_type = TypeOf(node->InputAt(0));
        Type* const rhs_type = TypeOf(node->InputAt(1));
        if (lhs_type->Is(Type::Unsigned32()) &&
            rhs_type->Is(Type::Unsigned32())) {
          VisitWord32TruncatingBinop(node);
          if (lower()) {
            lowering->DoMax(node, lowering->machine()->Uint32LessThan(),
                            MachineRepresentation::kWord32);
          }
        } else if (lhs_type->Is(Type::Signed32()) &&
                   rhs_type->Is(Type::Signed32())) {
          VisitWord32TruncatingBinop(node);
          if (lower()) {
            lowering->DoMax(node, lowering->machine()->Int32LessThan(),
                            MachineRepresentation::kWord32);
          }
        } else if (lhs_type->Is(Type::PlainNumber()) &&
                   rhs_type->Is(Type::PlainNumber())) {
          VisitFloat64Binop(node);
          if (lower()) {
            lowering->DoMax(node, lowering->machine()->Float64LessThan(),
                            MachineRepresentation::kFloat64);
          }
        } else {
          VisitFloat64Binop(node);
          if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        }
        return;
      }
      case IrOpcode::kNumberMin: {
        // It is safe to use the feedback types for left and right hand side
        // here, since we can only narrow those types and thus we can only
        // promise a more specific truncation.
        Type* const lhs_type = TypeOf(node->InputAt(0));
        Type* const rhs_type = TypeOf(node->InputAt(1));
        if (lhs_type->Is(Type::Unsigned32()) &&
            rhs_type->Is(Type::Unsigned32())) {
          VisitWord32TruncatingBinop(node);
          if (lower()) {
            lowering->DoMin(node, lowering->machine()->Uint32LessThan(),
                            MachineRepresentation::kWord32);
          }
        } else if (lhs_type->Is(Type::Signed32()) &&
                   rhs_type->Is(Type::Signed32())) {
          VisitWord32TruncatingBinop(node);
          if (lower()) {
            lowering->DoMin(node, lowering->machine()->Int32LessThan(),
                            MachineRepresentation::kWord32);
          }
        } else if (lhs_type->Is(Type::PlainNumber()) &&
                   rhs_type->Is(Type::PlainNumber())) {
          VisitFloat64Binop(node);
          if (lower()) {
            lowering->DoMin(node, lowering->machine()->Float64LessThan(),
                            MachineRepresentation::kFloat64);
          }
        } else {
          VisitFloat64Binop(node);
          if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        }
        return;
      }
      case IrOpcode::kNumberAtan2:
      case IrOpcode::kNumberPow: {
        VisitBinop(node, UseInfo::TruncatingFloat64(),
                   MachineRepresentation::kFloat64);
        if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kNumberAcos:
      case IrOpcode::kNumberAcosh:
      case IrOpcode::kNumberAsin:
      case IrOpcode::kNumberAsinh:
      case IrOpcode::kNumberAtan:
      case IrOpcode::kNumberAtanh:
      case IrOpcode::kNumberCeil:
      case IrOpcode::kNumberCos:
      case IrOpcode::kNumberCosh:
      case IrOpcode::kNumberExp:
      case IrOpcode::kNumberExpm1:
      case IrOpcode::kNumberFloor:
      case IrOpcode::kNumberLog:
      case IrOpcode::kNumberLog1p:
      case IrOpcode::kNumberLog2:
      case IrOpcode::kNumberLog10:
      case IrOpcode::kNumberCbrt:
      case IrOpcode::kNumberSin:
      case IrOpcode::kNumberSinh:
      case IrOpcode::kNumberTan:
      case IrOpcode::kNumberTanh:
      case IrOpcode::kNumberTrunc: {
        VisitUnop(node, UseInfo::TruncatingFloat64(),
                  MachineRepresentation::kFloat64);
        if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kNumberRound: {
        VisitUnop(node, UseInfo::TruncatingFloat64(),
                  MachineRepresentation::kFloat64);
        if (lower()) DeferReplacement(node, lowering->Float64Round(node));
        return;
      }
      case IrOpcode::kNumberSign: {
        if (InputIs(node, Type::Signed32())) {
          VisitUnop(node, UseInfo::TruncatingWord32(),
                    MachineRepresentation::kWord32);
          if (lower()) DeferReplacement(node, lowering->Int32Sign(node));
        } else {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kFloat64);
          if (lower()) DeferReplacement(node, lowering->Float64Sign(node));
        }
        return;
      }
      case IrOpcode::kNumberSqrt: {
        VisitUnop(node, UseInfo::TruncatingFloat64(),
                  MachineRepresentation::kFloat64);
        if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        return;
      }
      case IrOpcode::kNumberToBoolean: {
        Type* const input_type = TypeOf(node->InputAt(0));
        if (input_type->Is(Type::Integral32())) {
          VisitUnop(node, UseInfo::TruncatingWord32(),
                    MachineRepresentation::kBit);
          if (lower()) lowering->DoIntegral32ToBit(node);
        } else if (input_type->Is(Type::OrderedNumber())) {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kBit);
          if (lower()) lowering->DoOrderedNumberToBit(node);
        } else {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kBit);
          if (lower()) lowering->DoNumberToBit(node);
        }
        return;
      }
      case IrOpcode::kNumberToInt32: {
        // Just change representation if necessary.
        VisitUnop(node, UseInfo::TruncatingWord32(),
                  MachineRepresentation::kWord32);
        if (lower()) DeferReplacement(node, node->InputAt(0));
        return;
      }
      case IrOpcode::kNumberToString: {
        VisitUnop(node, UseInfo::AnyTagged(),
                  MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kNumberToUint32: {
        // Just change representation if necessary.
        VisitUnop(node, UseInfo::TruncatingWord32(),
                  MachineRepresentation::kWord32);
        if (lower()) DeferReplacement(node, node->InputAt(0));
        return;
      }
      case IrOpcode::kNumberToUint8Clamped: {
        Type* const input_type = TypeOf(node->InputAt(0));
        if (input_type->Is(type_cache_.kUint8OrMinusZeroOrNaN)) {
          VisitUnop(node, UseInfo::TruncatingWord32(),
                    MachineRepresentation::kWord32);
          if (lower()) DeferReplacement(node, node->InputAt(0));
        } else if (input_type->Is(Type::Unsigned32OrMinusZeroOrNaN())) {
          VisitUnop(node, UseInfo::TruncatingWord32(),
                    MachineRepresentation::kWord32);
          if (lower()) lowering->DoUnsigned32ToUint8Clamped(node);
        } else if (input_type->Is(Type::Signed32OrMinusZeroOrNaN())) {
          VisitUnop(node, UseInfo::TruncatingWord32(),
                    MachineRepresentation::kWord32);
          if (lower()) lowering->DoSigned32ToUint8Clamped(node);
        } else if (input_type->Is(type_cache_.kIntegerOrMinusZeroOrNaN)) {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kFloat64);
          if (lower()) lowering->DoIntegerToUint8Clamped(node);
        } else {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kFloat64);
          if (lower()) lowering->DoNumberToUint8Clamped(node);
        }
        return;
      }
      case IrOpcode::kReferenceEqual: {
        VisitBinop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        if (lower()) {
          NodeProperties::ChangeOp(node, lowering->machine()->WordEqual());
        }
        return;
      }
      case IrOpcode::kSameValue: {
        if (truncation.IsUnused()) return VisitUnused(node);
        VisitBinop(node, UseInfo::AnyTagged(),
                   MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kTypeOf: {
        return VisitUnop(node, UseInfo::AnyTagged(),
                         MachineRepresentation::kTaggedPointer);
      }
      case IrOpcode::kNewConsString: {
        ProcessInput(node, 0, UseInfo::TaggedSigned());  // length
        ProcessInput(node, 1, UseInfo::AnyTagged());     // first
        ProcessInput(node, 2, UseInfo::AnyTagged());     // second
        SetOutput(node, MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kStringEqual:
      case IrOpcode::kStringLessThan:
      case IrOpcode::kStringLessThanOrEqual: {
        return VisitBinop(node, UseInfo::AnyTagged(),
                          MachineRepresentation::kTaggedPointer);
      }
      case IrOpcode::kStringCharCodeAt: {
        return VisitBinop(node, UseInfo::AnyTagged(),
                          UseInfo::TruncatingWord32(),
                          MachineRepresentation::kWord32);
      }
      case IrOpcode::kStringCodePointAt: {
        return VisitBinop(node, UseInfo::AnyTagged(),
                          UseInfo::TruncatingWord32(),
                          MachineRepresentation::kTaggedSigned);
      }
      case IrOpcode::kStringFromCharCode: {
        VisitUnop(node, UseInfo::TruncatingWord32(),
                  MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kStringFromCodePoint: {
        VisitUnop(node, UseInfo::TruncatingWord32(),
                  MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kStringIndexOf: {
        ProcessInput(node, 0, UseInfo::AnyTagged());
        ProcessInput(node, 1, UseInfo::AnyTagged());
        ProcessInput(node, 2, UseInfo::TaggedSigned());
        SetOutput(node, MachineRepresentation::kTaggedSigned);
        return;
      }
      case IrOpcode::kStringLength: {
        // TODO(bmeurer): The input representation should be TaggedPointer.
        // Fix this once we have a dedicated StringConcat/JSStringAdd
        // operator, which marks it's output as TaggedPointer properly.
        VisitUnop(node, UseInfo::AnyTagged(),
                  MachineRepresentation::kTaggedSigned);
        return;
      }
      case IrOpcode::kStringSubstring: {
        ProcessInput(node, 0, UseInfo::AnyTagged());
        ProcessInput(node, 1, UseInfo::TruncatingWord32());
        ProcessInput(node, 2, UseInfo::TruncatingWord32());
        ProcessRemainingInputs(node, 3);
        SetOutput(node, MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kStringToLowerCaseIntl:
      case IrOpcode::kStringToUpperCaseIntl: {
        VisitUnop(node, UseInfo::AnyTagged(),
                  MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kCheckBounds: {
        const CheckParameters& p = CheckParametersOf(node->op());
        Type* index_type = TypeOf(node->InputAt(0));
        Type* length_type = TypeOf(node->InputAt(1));
        if (index_type->Is(Type::Integral32OrMinusZero())) {
          // Map -0 to 0, and the values in the [-2^31,-1] range to the
          // [2^31,2^32-1] range, which will be considered out-of-bounds
          // as well, because the {length_type} is limited to Unsigned31.
          VisitBinop(node, UseInfo::TruncatingWord32(),
                     MachineRepresentation::kWord32);
          if (lower()) {
            if (index_type->IsNone() || length_type->IsNone() ||
                (index_type->Min() >= 0.0 &&
                 index_type->Max() < length_type->Min())) {
              // The bounds check is redundant if we already know that
              // the index is within the bounds of [0.0, length[.
              DeferReplacement(node, node->InputAt(0));
            }
          }
        } else {
          VisitBinop(
              node,
              UseInfo::CheckedSigned32AsWord32(kIdentifyZeros, p.feedback()),
              UseInfo::TruncatingWord32(), MachineRepresentation::kWord32);
        }
        return;
      }
      case IrOpcode::kMaskIndexWithBound: {
        VisitBinop(node, UseInfo::TruncatingWord32(),
                   MachineRepresentation::kWord32);
        return;
      }
      case IrOpcode::kCheckHeapObject: {
        if (InputCannotBe(node, Type::SignedSmall())) {
          VisitUnop(node, UseInfo::AnyTagged(),
                    MachineRepresentation::kTaggedPointer);
        } else {
          VisitUnop(node, UseInfo::CheckedHeapObjectAsTaggedPointer(),
                    MachineRepresentation::kTaggedPointer);
        }
        if (lower()) DeferReplacement(node, node->InputAt(0));
        return;
      }
      case IrOpcode::kCheckIf: {
        ProcessInput(node, 0, UseInfo::Bool());
        ProcessRemainingInputs(node, 1);
        SetOutput(node, MachineRepresentation::kNone);
        return;
      }
      case IrOpcode::kCheckInternalizedString: {
        VisitCheck(node, Type::InternalizedString(), lowering);
        return;
      }
      case IrOpcode::kCheckNumber: {
        Type* const input_type = TypeOf(node->InputAt(0));
        if (input_type->Is(Type::Number())) {
          VisitNoop(node, truncation);
        } else {
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kTagged);
        }
        return;
      }
      case IrOpcode::kCheckReceiver: {
        VisitCheck(node, Type::Receiver(), lowering);
        return;
      }
      case IrOpcode::kCheckSmi: {
        const CheckParameters& params = CheckParametersOf(node->op());
        if (SmiValuesAre32Bits() && truncation.IsUsedAsWord32()) {
          VisitUnop(node,
                    UseInfo::CheckedSignedSmallAsWord32(kDistinguishZeros,
                                                        params.feedback()),
                    MachineRepresentation::kWord32);
        } else {
          VisitUnop(
              node,
              UseInfo::CheckedSignedSmallAsTaggedSigned(params.feedback()),
              MachineRepresentation::kTaggedSigned);
        }
        if (lower()) DeferReplacement(node, node->InputAt(0));
        return;
      }
      case IrOpcode::kCheckString: {
        VisitCheck(node, Type::String(), lowering);
        return;
      }
      case IrOpcode::kCheckSymbol: {
        VisitCheck(node, Type::Symbol(), lowering);
        return;
      }

      case IrOpcode::kAllocate: {
        ProcessInput(node, 0, UseInfo::TruncatingWord32());
        ProcessRemainingInputs(node, 1);
        SetOutput(node, MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kLoadFieldByIndex: {
        if (truncation.IsUnused()) return VisitUnused(node);
        VisitBinop(node, UseInfo::AnyTagged(), UseInfo::TruncatingWord32(),
                   MachineRepresentation::kTagged);
        return;
      }
      case IrOpcode::kLoadField: {
        if (truncation.IsUnused()) return VisitUnused(node);
        FieldAccess access = FieldAccessOf(node->op());
        MachineRepresentation const representation =
            access.machine_type.representation();
        VisitUnop(node, UseInfoForBasePointer(access), representation);
        return;
      }
      case IrOpcode::kStoreField: {
        FieldAccess access = FieldAccessOf(node->op());
        Node* value_node = node->InputAt(1);
        NodeInfo* input_info = GetInfo(value_node);
        MachineRepresentation field_representation =
            access.machine_type.representation();

        // Convert to Smi if possible, such that we can avoid a write barrier.
        if (field_representation == MachineRepresentation::kTagged &&
            TypeOf(value_node)->Is(Type::SignedSmall())) {
          field_representation = MachineRepresentation::kTaggedSigned;
        }
        WriteBarrierKind write_barrier_kind = WriteBarrierKindFor(
            access.base_is_tagged, field_representation, access.offset,
            access.type, input_info->representation(), value_node);

        ProcessInput(node, 0, UseInfoForBasePointer(access));
        ProcessInput(node, 1,
                     TruncatingUseInfoFromRepresentation(field_representation));
        ProcessRemainingInputs(node, 2);
        SetOutput(node, MachineRepresentation::kNone);
        if (lower()) {
          if (write_barrier_kind < access.write_barrier_kind) {
            access.write_barrier_kind = write_barrier_kind;
            NodeProperties::ChangeOp(
                node, jsgraph_->simplified()->StoreField(access));
          }
        }
        return;
      }
      case IrOpcode::kLoadElement: {
        if (truncation.IsUnused()) return VisitUnused(node);
        ElementAccess access = ElementAccessOf(node->op());
        VisitBinop(node, UseInfoForBasePointer(access),
                   UseInfo::TruncatingWord32(),
                   access.machine_type.representation());
        return;
      }
      case IrOpcode::kStoreElement: {
        ElementAccess access = ElementAccessOf(node->op());
        Node* value_node = node->InputAt(2);
        NodeInfo* input_info = GetInfo(value_node);
        MachineRepresentation element_representation =
            access.machine_type.representation();

        // Convert to Smi if possible, such that we can avoid a write barrier.
        if (element_representation == MachineRepresentation::kTagged &&
            TypeOf(value_node)->Is(Type::SignedSmall())) {
          element_representation = MachineRepresentation::kTaggedSigned;
        }
        WriteBarrierKind write_barrier_kind = WriteBarrierKindFor(
            access.base_is_tagged, element_representation, access.type,
            input_info->representation(), value_node);
        ProcessInput(node, 0, UseInfoForBasePointer(access));  // base
        ProcessInput(node, 1, UseInfo::TruncatingWord32());    // index
        ProcessInput(node, 2,
                     TruncatingUseInfoFromRepresentation(
                         element_representation));  // value
        ProcessRemainingInputs(node, 3);
        SetOutput(node, MachineRepresentation::kNone);
        if (lower()) {
          if (write_barrier_kind < access.write_barrier_kind) {
            access.write_barrier_kind = write_barrier_kind;
            NodeProperties::ChangeOp(
                node, jsgraph_->simplified()->StoreElement(access));
          }
        }
        return;
      }
      case IrOpcode::kNumberIsFloat64Hole: {
        VisitUnop(node, UseInfo::TruncatingFloat64(),
                  MachineRepresentation::kBit);
        return;
      }
      case IrOpcode::kTransitionAndStoreElement: {
        Type* value_type = TypeOf(node->InputAt(2));

        ProcessInput(node, 0, UseInfo::AnyTagged());         // array
        ProcessInput(node, 1, UseInfo::TruncatingWord32());  // index

        if (value_type->Is(Type::SignedSmall())) {
          ProcessInput(node, 2, UseInfo::TruncatingWord32());  // value
          if (lower()) {
            NodeProperties::ChangeOp(node,
                                     simplified()->StoreSignedSmallElement());
          }
        } else if (value_type->Is(Type::Number())) {
          ProcessInput(node, 2, UseInfo::TruncatingFloat64());  // value
          if (lower()) {
            Handle<Map> double_map = DoubleMapParameterOf(node->op());
            NodeProperties::ChangeOp(
                node,
                simplified()->TransitionAndStoreNumberElement(double_map));
          }
        } else if (value_type->Is(Type::NonNumber())) {
          ProcessInput(node, 2, UseInfo::AnyTagged());  // value
          if (lower()) {
            Handle<Map> fast_map = FastMapParameterOf(node->op());
            NodeProperties::ChangeOp(
                node, simplified()->TransitionAndStoreNonNumberElement(
                          fast_map, value_type));
          }
        } else {
          ProcessInput(node, 2, UseInfo::AnyTagged());  // value
        }

        ProcessRemainingInputs(node, 3);
        SetOutput(node, MachineRepresentation::kNone);
        return;
      }
      case IrOpcode::kLoadTypedElement: {
        MachineRepresentation const rep =
            MachineRepresentationFromArrayType(ExternalArrayTypeOf(node->op()));
        ProcessInput(node, 0, UseInfo::AnyTagged());         // buffer
        ProcessInput(node, 1, UseInfo::AnyTagged());         // base pointer
        ProcessInput(node, 2, UseInfo::PointerInt());        // external pointer
        ProcessInput(node, 3, UseInfo::TruncatingWord32());  // index
        ProcessRemainingInputs(node, 4);
        SetOutput(node, rep);
        return;
      }
      case IrOpcode::kStoreTypedElement: {
        MachineRepresentation const rep =
            MachineRepresentationFromArrayType(ExternalArrayTypeOf(node->op()));
        ProcessInput(node, 0, UseInfo::AnyTagged());         // buffer
        ProcessInput(node, 1, UseInfo::AnyTagged());         // base pointer
        ProcessInput(node, 2, UseInfo::PointerInt());        // external pointer
        ProcessInput(node, 3, UseInfo::TruncatingWord32());  // index
        ProcessInput(node, 4,
                     TruncatingUseInfoFromRepresentation(rep));  // value
        ProcessRemainingInputs(node, 5);
        SetOutput(node, MachineRepresentation::kNone);
        return;
      }
      case IrOpcode::kConvertReceiver: {
        Type* input_type = TypeOf(node->InputAt(0));
        VisitBinop(node, UseInfo::AnyTagged(),
                   MachineRepresentation::kTaggedPointer);
        if (lower()) {
          // Try to optimize the {node} based on the input type.
          if (input_type->Is(Type::Receiver())) {
            DeferReplacement(node, node->InputAt(0));
          } else if (input_type->Is(Type::NullOrUndefined())) {
            DeferReplacement(node, node->InputAt(1));
          } else if (!input_type->Maybe(Type::NullOrUndefined())) {
            NodeProperties::ChangeOp(
                node, lowering->simplified()->ConvertReceiver(
                          ConvertReceiverMode::kNotNullOrUndefined));
          }
        }
        return;
      }
      case IrOpcode::kPlainPrimitiveToNumber: {
        if (InputIs(node, Type::Boolean())) {
          VisitUnop(node, UseInfo::Bool(), MachineRepresentation::kWord32);
          if (lower()) DeferReplacement(node, node->InputAt(0));
        } else if (InputIs(node, Type::String())) {
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kTagged);
          if (lower()) {
            NodeProperties::ChangeOp(node, simplified()->StringToNumber());
          }
        } else if (truncation.IsUsedAsWord32()) {
          if (InputIs(node, Type::NumberOrOddball())) {
            VisitUnop(node, UseInfo::TruncatingWord32(),
                      MachineRepresentation::kWord32);
            if (lower()) DeferReplacement(node, node->InputAt(0));
          } else {
            VisitUnop(node, UseInfo::AnyTagged(),
                      MachineRepresentation::kWord32);
            if (lower()) {
              NodeProperties::ChangeOp(node,
                                       simplified()->PlainPrimitiveToWord32());
            }
          }
        } else if (truncation.IsUsedAsFloat64()) {
          if (InputIs(node, Type::NumberOrOddball())) {
            VisitUnop(node, UseInfo::TruncatingFloat64(),
                      MachineRepresentation::kFloat64);
            if (lower()) DeferReplacement(node, node->InputAt(0));
          } else {
            VisitUnop(node, UseInfo::AnyTagged(),
                      MachineRepresentation::kFloat64);
            if (lower()) {
              NodeProperties::ChangeOp(node,
                                       simplified()->PlainPrimitiveToFloat64());
            }
          }
        } else {
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kTagged);
        }
        return;
      }
      case IrOpcode::kSpeculativeToNumber: {
        NumberOperationParameters const& p =
            NumberOperationParametersOf(node->op());
        switch (p.hint()) {
          case NumberOperationHint::kSigned32:
          case NumberOperationHint::kSignedSmall:
          case NumberOperationHint::kSignedSmallInputs:
            VisitUnop(node,
                      CheckedUseInfoAsWord32FromHint(p.hint(), p.feedback()),
                      MachineRepresentation::kWord32, Type::Signed32());
            break;
          case NumberOperationHint::kNumber:
          case NumberOperationHint::kNumberOrOddball:
            VisitUnop(node,
                      CheckedUseInfoAsFloat64FromHint(p.hint(), p.feedback()),
                      MachineRepresentation::kFloat64);
            break;
        }
        if (lower()) DeferReplacement(node, node->InputAt(0));
        return;
      }
      case IrOpcode::kObjectIsArrayBufferView: {
        // TODO(turbofan): Introduce a Type::ArrayBufferView?
        VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        return;
      }
      case IrOpcode::kObjectIsBigInt: {
        VisitObjectIs(node, Type::BigInt(), lowering);
        return;
      }
      case IrOpcode::kObjectIsCallable: {
        VisitObjectIs(node, Type::Callable(), lowering);
        return;
      }
      case IrOpcode::kObjectIsConstructor: {
        // TODO(turbofan): Introduce a Type::Constructor?
        VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        return;
      }
      case IrOpcode::kObjectIsDetectableCallable: {
        VisitObjectIs(node, Type::DetectableCallable(), lowering);
        return;
      }
      case IrOpcode::kObjectIsFiniteNumber: {
        Type* const input_type = GetUpperBound(node->InputAt(0));
        if (input_type->Is(type_cache_.kSafeInteger)) {
          VisitUnop(node, UseInfo::None(), MachineRepresentation::kBit);
          if (lower()) {
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(1));
          }
        } else if (!input_type->Maybe(Type::Number())) {
          VisitUnop(node, UseInfo::Any(), MachineRepresentation::kBit);
          if (lower()) {
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(0));
          }
        } else if (input_type->Is(Type::Number())) {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kBit);
          if (lower()) {
            NodeProperties::ChangeOp(node,
                                     lowering->simplified()->NumberIsFinite());
          }
        } else {
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        }
        return;
      }
      case IrOpcode::kNumberIsFinite: {
        UNREACHABLE();
      }
      case IrOpcode::kObjectIsInteger: {
        Type* const input_type = GetUpperBound(node->InputAt(0));
        if (input_type->Is(type_cache_.kSafeInteger)) {
          VisitUnop(node, UseInfo::None(), MachineRepresentation::kBit);
          if (lower()) {
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(1));
          }
        } else if (!input_type->Maybe(Type::Number())) {
          VisitUnop(node, UseInfo::Any(), MachineRepresentation::kBit);
          if (lower()) {
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(0));
          }
        } else if (input_type->Is(Type::Number())) {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kBit);
          if (lower()) {
            NodeProperties::ChangeOp(node,
                                     lowering->simplified()->NumberIsInteger());
          }
        } else {
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        }
        return;
      }
      case IrOpcode::kNumberIsInteger: {
        VisitUnop(node, UseInfo::TruncatingFloat64(),
                  MachineRepresentation::kBit);
        return;
      }
      case IrOpcode::kObjectIsMinusZero: {
        Type* const input_type = GetUpperBound(node->InputAt(0));
        if (input_type->Is(Type::MinusZero())) {
          VisitUnop(node, UseInfo::None(), MachineRepresentation::kBit);
          if (lower()) {
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(1));
          }
        } else if (!input_type->Maybe(Type::MinusZero())) {
          VisitUnop(node, UseInfo::Any(), MachineRepresentation::kBit);
          if (lower()) {
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(0));
          }
        } else if (input_type->Is(Type::Number())) {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kBit);
          if (lower()) {
            // ObjectIsMinusZero(x:kRepFloat64)
            //   => Float64Equal(Float64Div(1.0,x),-Infinity)
            Node* const input = node->InputAt(0);
            node->ReplaceInput(
                0, jsgraph_->graph()->NewNode(
                       lowering->machine()->Float64Div(),
                       lowering->jsgraph()->Float64Constant(1.0), input));
            node->AppendInput(jsgraph_->zone(),
                              jsgraph_->Float64Constant(
                                  -std::numeric_limits<double>::infinity()));
            NodeProperties::ChangeOp(node, lowering->machine()->Float64Equal());
          }
        } else {
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        }
        return;
      }
      case IrOpcode::kObjectIsNaN: {
        Type* const input_type = GetUpperBound(node->InputAt(0));
        if (input_type->Is(Type::NaN())) {
          VisitUnop(node, UseInfo::None(), MachineRepresentation::kBit);
          if (lower()) {
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(1));
          }
        } else if (!input_type->Maybe(Type::NaN())) {
          VisitUnop(node, UseInfo::Any(), MachineRepresentation::kBit);
          if (lower()) {
            DeferReplacement(node, lowering->jsgraph()->Int32Constant(0));
          }
        } else if (input_type->Is(Type::Number())) {
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kBit);
          if (lower()) {
            // ObjectIsNaN(x:kRepFloat64) => Word32Equal(Float64Equal(x,x),#0)
            Node* const input = node->InputAt(0);
            node->ReplaceInput(
                0, jsgraph_->graph()->NewNode(
                       lowering->machine()->Float64Equal(), input, input));
            node->AppendInput(jsgraph_->zone(), jsgraph_->Int32Constant(0));
            NodeProperties::ChangeOp(node, lowering->machine()->Word32Equal());
          }
        } else {
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        }
        return;
      }
      case IrOpcode::kObjectIsNonCallable: {
        VisitObjectIs(node, Type::NonCallable(), lowering);
        return;
      }
      case IrOpcode::kObjectIsNumber: {
        VisitObjectIs(node, Type::Number(), lowering);
        return;
      }
      case IrOpcode::kObjectIsReceiver: {
        VisitObjectIs(node, Type::Receiver(), lowering);
        return;
      }
      case IrOpcode::kObjectIsSmi: {
        // TODO(turbofan): Optimize based on input representation.
        VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        return;
      }
      case IrOpcode::kObjectIsString: {
        VisitObjectIs(node, Type::String(), lowering);
        return;
      }
      case IrOpcode::kObjectIsSymbol: {
        VisitObjectIs(node, Type::Symbol(), lowering);
        return;
      }
      case IrOpcode::kObjectIsUndetectable: {
        VisitObjectIs(node, Type::Undetectable(), lowering);
        return;
      }
      case IrOpcode::kArgumentsFrame: {
        SetOutput(node, MachineType::PointerRepresentation());
        return;
      }
      case IrOpcode::kArgumentsLength: {
        VisitUnop(node, UseInfo::PointerInt(),
                  MachineRepresentation::kTaggedSigned);
        return;
      }
      case IrOpcode::kNewDoubleElements:
      case IrOpcode::kNewSmiOrObjectElements: {
        VisitUnop(node, UseInfo::TruncatingWord32(),
                  MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kNewArgumentsElements: {
        VisitBinop(node, UseInfo::PointerInt(), UseInfo::TaggedSigned(),
                   MachineRepresentation::kTaggedPointer);
        return;
      }
      case IrOpcode::kArrayBufferWasNeutered: {
        VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kBit);
        return;
      }
      case IrOpcode::kCheckFloat64Hole: {
        Type* const input_type = TypeOf(node->InputAt(0));
        if (input_type->Is(Type::Number())) {
          VisitNoop(node, truncation);
        } else {
          CheckFloat64HoleMode mode = CheckFloat64HoleModeOf(node->op());
          switch (mode) {
            case CheckFloat64HoleMode::kAllowReturnHole:
              if (truncation.IsUnused()) return VisitUnused(node);
              if (truncation.IsUsedAsFloat64()) {
                VisitUnop(node, UseInfo::TruncatingFloat64(),
                          MachineRepresentation::kFloat64);
                if (lower()) DeferReplacement(node, node->InputAt(0));
              } else {
                VisitUnop(
                    node,
                    UseInfo(MachineRepresentation::kFloat64, Truncation::Any()),
                    MachineRepresentation::kFloat64, Type::Number());
              }
              break;
            case CheckFloat64HoleMode::kNeverReturnHole:
              VisitUnop(
                  node,
                  UseInfo(MachineRepresentation::kFloat64, Truncation::Any()),
                  MachineRepresentation::kFloat64, Type::Number());
              break;
          }
        }
        return;
      }
      case IrOpcode::kCheckNotTaggedHole: {
        VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kTagged);
        return;
      }
      case IrOpcode::kConvertTaggedHoleToUndefined: {
        if (InputIs(node, Type::NumberOrOddball()) &&
            truncation.IsUsedAsWord32()) {
          // Propagate the Word32 truncation.
          VisitUnop(node, UseInfo::TruncatingWord32(),
                    MachineRepresentation::kWord32);
          if (lower()) DeferReplacement(node, node->InputAt(0));
        } else if (InputIs(node, Type::NumberOrOddball()) &&
                   truncation.IsUsedAsFloat64()) {
          // Propagate the Float64 truncation.
          VisitUnop(node, UseInfo::TruncatingFloat64(),
                    MachineRepresentation::kFloat64);
          if (lower()) DeferReplacement(node, node->InputAt(0));
        } else if (InputIs(node, Type::NonInternal())) {
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kTagged);
          if (lower()) DeferReplacement(node, node->InputAt(0));
        } else {
          // TODO(turbofan): Add a (Tagged) truncation that identifies hole
          // and undefined, i.e. for a[i] === obj cases.
          VisitUnop(node, UseInfo::AnyTagged(), MachineRepresentation::kTagged);
        }
        return;
      }
      case IrOpcode::kCheckEqualsSymbol:
      case IrOpcode::kCheckEqualsInternalizedString:
        return VisitBinop(node, UseInfo::AnyTagged(),
                          MachineRepresentation::kNone);
      case IrOpcode::kMapGuard:
        // Eliminate MapGuard nodes here.
        return VisitUnused(node);
      case IrOpcode::kCheckMaps:
      case IrOpcode::kTransitionElementsKind: {
        VisitInputs(node);
        return SetOutput(node, MachineRepresentation::kNone);
      }
      case IrOpcode::kCompareMaps:
        return VisitUnop(node, UseInfo::AnyTagged(),
                         MachineRepresentation::kBit);
      case IrOpcode::kEnsureWritableFastElements:
        return VisitBinop(node, UseInfo::AnyTagged(),
                          MachineRepresentation::kTaggedPointer);
      case IrOpcode::kMaybeGrowFastElements: {
        ProcessInput(node, 0, UseInfo::AnyTagged());         // object
        ProcessInput(node, 1, UseInfo::AnyTagged());         // elements
        ProcessInput(node, 2, UseInfo::TruncatingWord32());  // index
        ProcessInput(node, 3, UseInfo::TruncatingWord32());  // length
        ProcessRemainingInputs(node, 4);
        SetOutput(node, MachineRepresentation::kTaggedPointer);
        return;
      }

      case IrOpcode::kNumberSilenceNaN:
        VisitUnop(node, UseInfo::TruncatingFloat64(),
                  MachineRepresentation::kFloat64);
        if (lower()) NodeProperties::ChangeOp(node, Float64Op(node));
        return;
      case IrOpcode::kFrameState:
        return VisitFrameState(node);
      case IrOpcode::kStateValues:
        return VisitStateValues(node);
      case IrOpcode::kObjectState:
        return VisitObjectState(node);
      case IrOpcode::kObjectId:
        return SetOutput(node, MachineRepresentation::kTaggedPointer);
      case IrOpcode::kTypeGuard: {
        // We just get rid of the sigma here, choosing the best representation
        // for the sigma's type.
        Type* type = TypeOf(node);
        MachineRepresentation representation =
            GetOutputInfoForPhi(node, type, truncation);

        // Here we pretend that the input has the sigma's type for the
        // conversion.
        UseInfo use(representation, truncation);
        if (propagate()) {
          EnqueueInput(node, 0, use);
        } else if (lower()) {
          ConvertInput(node, 0, use, type);
        }
        ProcessRemainingInputs(node, 1);
        SetOutput(node, representation);
        return;
      }

      case IrOpcode::kFinishRegion:
        VisitInputs(node);
        // Assume the output is tagged pointer.
        return SetOutput(node, MachineRepresentation::kTaggedPointer);

      case IrOpcode::kReturn:
        VisitReturn(node);
        // Assume the output is tagged.
        return SetOutput(node, MachineRepresentation::kTagged);

      case IrOpcode::kFindOrderedHashMapEntry: {
        Type* const key_type = TypeOf(node->InputAt(1));
        if (key_type->Is(Type::Signed32OrMinusZero())) {
          VisitBinop(node, UseInfo::AnyTagged(), UseInfo::TruncatingWord32(),
                     MachineRepresentation::kWord32);
          if (lower()) {
            NodeProperties::ChangeOp(
                node,
                lowering->simplified()->FindOrderedHashMapEntryForInt32Key());
          }
        } else {
          VisitBinop(node, UseInfo::AnyTagged(),
                     MachineRepresentation::kTaggedSigned);
        }
        return;
      }

      // Operators with all inputs tagged and no or tagged output have uniform
      // handling.
      case IrOpcode::kEnd:
      case IrOpcode::kIfSuccess:
      case IrOpcode::kIfException:
      case IrOpcode::kIfTrue:
      case IrOpcode::kIfFalse:
      case IrOpcode::kIfValue:
      case IrOpcode::kIfDefault:
      case IrOpcode::kDeoptimize:
      case IrOpcode::kEffectPhi:
      case IrOpcode::kTerminate:
      case IrOpcode::kCheckpoint:
      case IrOpcode::kLoop:
      case IrOpcode::kMerge:
      case IrOpcode::kThrow:
      case IrOpcode::kBeginRegion:
      case IrOpcode::kProjection:
      case IrOpcode::kOsrValue:
      case IrOpcode::kArgumentsElementsState:
      case IrOpcode::kArgumentsLengthState:
      case IrOpcode::kUnreachable:
      case IrOpcode::kRuntimeAbort:
// All JavaScript operators except JSToNumber have uniform handling.
#define OPCODE_CASE(name) case IrOpcode::k##name:
        JS_SIMPLE_BINOP_LIST(OPCODE_CASE)
        JS_OBJECT_OP_LIST(OPCODE_CASE)
        JS_CONTEXT_OP_LIST(OPCODE_CASE)
        JS_OTHER_OP_LIST(OPCODE_CASE)
#undef OPCODE_CASE
      case IrOpcode::kJSBitwiseNot:
      case IrOpcode::kJSDecrement:
      case IrOpcode::kJSIncrement:
      case IrOpcode::kJSNegate:
      case IrOpcode::kJSToInteger:
      case IrOpcode::kJSToLength:
      case IrOpcode::kJSToName:
      case IrOpcode::kJSToObject:
      case IrOpcode::kJSToString:
        VisitInputs(node);
        // Assume the output is tagged.
        return SetOutput(node, MachineRepresentation::kTagged);
      case IrOpcode::kDeadValue:
        ProcessInput(node, 0, UseInfo::Any());
        return SetOutput(node, MachineRepresentation::kNone);
      default:
        FATAL(
            "Representation inference: unsupported opcode %i (%s), node #%i\n.",
            node->opcode(), node->op()->mnemonic(), node->id());
        break;
    }
    UNREACHABLE();
  }
