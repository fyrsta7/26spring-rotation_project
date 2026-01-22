class MachineLoweringReducer : public Next {
 public:
  TURBOSHAFT_REDUCER_BOILERPLATE()

  bool NeedsHeapObjectCheck(ObjectIsOp::InputAssumptions input_assumptions) {
    // TODO(nicohartmann@): Consider type information once we have that.
    switch (input_assumptions) {
      case ObjectIsOp::InputAssumptions::kNone:
        return true;
      case ObjectIsOp::InputAssumptions::kHeapObject:
      case ObjectIsOp::InputAssumptions::kBigInt:
        return false;
    }
  }

  OpIndex REDUCE(ChangeOrDeopt)(OpIndex input, OpIndex frame_state,
                                ChangeOrDeoptOp::Kind kind,
                                CheckForMinusZeroMode minus_zero_mode,
                                const FeedbackSource& feedback) {
    switch (kind) {
      case ChangeOrDeoptOp::Kind::kUint32ToInt32: {
        __ DeoptimizeIf(__ Int32LessThan(input, 0), frame_state,
                        DeoptimizeReason::kLostPrecision, feedback);
        return input;
      }
      case ChangeOrDeoptOp::Kind::kInt64ToInt32: {
        V<Word32> i32 = __ TruncateWord64ToWord32(input);
        __ DeoptimizeIfNot(__ Word64Equal(__ ChangeInt32ToInt64(i32), input),
                           frame_state, DeoptimizeReason::kLostPrecision,
                           feedback);
        return i32;
      }
      case ChangeOrDeoptOp::Kind::kUint64ToInt32: {
        __ DeoptimizeIfNot(
            __ Uint64LessThanOrEqual(input, static_cast<uint64_t>(kMaxInt)),
            frame_state, DeoptimizeReason::kLostPrecision, feedback);
        return __ TruncateWord64ToWord32(input);
      }
      case ChangeOrDeoptOp::Kind::kUint64ToInt64: {
        __ DeoptimizeIfNot(__ Uint64LessThanOrEqual(
                               input, std::numeric_limits<int64_t>::max()),
                           frame_state, DeoptimizeReason::kLostPrecision,
                           feedback);
        return input;
      }
      case ChangeOrDeoptOp::Kind::kFloat64ToInt32: {
        V<Word32> i32 = __ TruncateFloat64ToInt32OverflowUndefined(input);
        __ DeoptimizeIfNot(__ Float64Equal(__ ChangeInt32ToFloat64(i32), input),
                           frame_state, DeoptimizeReason::kLostPrecisionOrNaN,
                           feedback);

        if (minus_zero_mode == CheckForMinusZeroMode::kCheckForMinusZero) {
          // Check if {value} is -0.
          IF(UNLIKELY(__ Word32Equal(i32, 0))) {
            // In case of 0, we need to check the high bits for the IEEE -0
            // pattern.
            V<Word32> check_negative =
                __ Int32LessThan(__ Float64ExtractHighWord32(input), 0);
            __ DeoptimizeIf(check_negative, frame_state,
                            DeoptimizeReason::kMinusZero, feedback);
          }
          END_IF
        }

        return i32;
      }
      case ChangeOrDeoptOp::Kind::kFloat64ToInt64: {
        V<Word64> i64 = __ TruncateFloat64ToInt64OverflowToMin(input);
        __ DeoptimizeIfNot(__ Float64Equal(__ ChangeInt64ToFloat64(i64), input),
                           frame_state, DeoptimizeReason::kLostPrecisionOrNaN,
                           feedback);

        if (minus_zero_mode == CheckForMinusZeroMode::kCheckForMinusZero) {
          // Check if {value} is -0.
          IF(UNLIKELY(__ Word64Equal(i64, 0))) {
            // In case of 0, we need to check the high bits for the IEEE -0
            // pattern.
            V<Word32> check_negative =
                __ Int32LessThan(__ Float64ExtractHighWord32(input), 0);
            __ DeoptimizeIf(check_negative, frame_state,
                            DeoptimizeReason::kMinusZero, feedback);
          }
          END_IF
        }

        return i64;
      }
      case ChangeOrDeoptOp::Kind::kFloat64NotHole: {
        // First check whether {value} is a NaN at all...
        IF_NOT (LIKELY(__ Float64Equal(input, input))) {
          // ...and only if {value} is a NaN, perform the expensive bit
          // check. See http://crbug.com/v8/8264 for details.
          __ DeoptimizeIf(__ Word32Equal(__ Float64ExtractHighWord32(input),
                                         kHoleNanUpper32),
                          frame_state, DeoptimizeReason::kHole, feedback);
        }
        END_IF
        return input;
      }
    }
    UNREACHABLE();
  }

  OpIndex REDUCE(DeoptimizeIf)(OpIndex condition, OpIndex frame_state,
                               bool negated,
                               const DeoptimizeParameters* parameters) {
    LABEL_BLOCK(no_change) {
      return Next::ReduceDeoptimizeIf(condition, frame_state, negated,
                                      parameters);
    }
    if (ShouldSkipOptimizationStep()) goto no_change;
    // Block cloning only works for branches, but not for `DeoptimizeIf`. On the
    // other hand, explicit control flow makes the overall pipeline and
    // escpecially the register allocator slower. So we only switch a
    // `DeoptiomizeIf` to a branch if it has a phi input, which indicates that
    // block cloning could be helpful.
    if (__ Get(condition).template Is<PhiOp>()) {
      if (negated) {
        IF_NOT (LIKELY(condition)) {
          __ Deoptimize(frame_state, parameters);
        }
        END_IF
      } else {
        IF (UNLIKELY(condition)) {
          __ Deoptimize(frame_state, parameters);
        }
        END_IF
      }
      return OpIndex::Invalid();
    }
    goto no_change;
  }

  V<Word32> REDUCE(ObjectIs)(V<Tagged> input, ObjectIsOp::Kind kind,
                             ObjectIsOp::InputAssumptions input_assumptions) {
    switch (kind) {
      case ObjectIsOp::Kind::kBigInt:
      case ObjectIsOp::Kind::kBigInt64: {
        DCHECK_IMPLIES(kind == ObjectIsOp::Kind::kBigInt64, Is64());

        Label<Word32> done(this);

        if (input_assumptions != ObjectIsOp::InputAssumptions::kBigInt) {
          if (NeedsHeapObjectCheck(input_assumptions)) {
            // Check for Smi.
            GOTO_IF(__ IsSmi(input), done, 0);
          }

          // Check for BigInt.
          V<Map> map = __ LoadMapField(input);
          V<Word32> is_bigint_map =
              __ TaggedEqual(map, __ HeapConstant(factory_->bigint_map()));
          GOTO_IF_NOT(is_bigint_map, done, 0);
        }

        if (kind == ObjectIsOp::Kind::kBigInt) {
          GOTO(done, 1);
        } else {
          DCHECK_EQ(kind, ObjectIsOp::Kind::kBigInt64);
          // We have to perform check for BigInt64 range.
          V<Word32> bitfield = __ template LoadField<Word32>(
              input, AccessBuilder::ForBigIntBitfield());
          GOTO_IF(__ Word32Equal(bitfield, 0), done, 1);

          // Length must be 1.
          V<Word32> length_field =
              __ Word32BitwiseAnd(bitfield, BigInt::LengthBits::kMask);
          GOTO_IF_NOT(__ Word32Equal(length_field,
                                     uint32_t{1} << BigInt::LengthBits::kShift),
                      done, 0);

          // Check if it fits in 64 bit signed int.
          V<Word64> lsd = __ template LoadField<Word64>(
              input, AccessBuilder::ForBigIntLeastSignificantDigit64());
          V<Word32> magnitude_check = __ Uint64LessThanOrEqual(
              lsd, std::numeric_limits<int64_t>::max());
          GOTO_IF(magnitude_check, done, 1);

          // The BigInt probably doesn't fit into signed int64. The only
          // exception is int64_t::min. We check for this.
          V<Word32> sign =
              __ Word32BitwiseAnd(bitfield, BigInt::SignBits::kMask);
          V<Word32> sign_check = __ Word32Equal(sign, BigInt::SignBits::kMask);
          GOTO_IF_NOT(sign_check, done, 0);

          V<Word32> min_check =
              __ Word64Equal(lsd, std::numeric_limits<int64_t>::min());
          GOTO_IF(min_check, done, 1);

          GOTO(done, 0);
        }

        BIND(done, result);
        return result;
      }
      case ObjectIsOp::Kind::kUndetectable:
        if (DependOnNoUndetectableObjectsProtector()) {
          V<Word32> is_undefined = __ TaggedEqual(
              input, __ HeapConstant(factory_->undefined_value()));
          V<Word32> is_null =
              __ TaggedEqual(input, __ HeapConstant(factory_->null_value()));
          return __ Word32BitwiseOr(is_undefined, is_null);
        }
        V8_FALLTHROUGH;
      case ObjectIsOp::Kind::kCallable:
      case ObjectIsOp::Kind::kConstructor:
      case ObjectIsOp::Kind::kDetectableCallable:
      case ObjectIsOp::Kind::kNonCallable:
      case ObjectIsOp::Kind::kReceiver:
      case ObjectIsOp::Kind::kReceiverOrNullOrUndefined: {
        Label<Word32> done(this);

        // Check for Smi if necessary.
        if (NeedsHeapObjectCheck(input_assumptions)) {
          GOTO_IF(UNLIKELY(__ IsSmi(input)), done, 0);
        }

        // Load bitfield from map.
        V<Map> map = __ LoadMapField(input);
        V<Word32> bitfield =
            __ template LoadField<Word32>(map, AccessBuilder::ForMapBitField());

        V<Word32> check;
        switch (kind) {
          case ObjectIsOp::Kind::kCallable:
            check =
                __ Word32Equal(Map::Bits1::IsCallableBit::kMask,
                               __ Word32BitwiseAnd(
                                   bitfield, Map::Bits1::IsCallableBit::kMask));
            break;
          case ObjectIsOp::Kind::kConstructor:
            check = __ Word32Equal(
                Map::Bits1::IsConstructorBit::kMask,
                __ Word32BitwiseAnd(bitfield,
                                    Map::Bits1::IsConstructorBit::kMask));
            break;
          case ObjectIsOp::Kind::kDetectableCallable:
            check = __ Word32Equal(
                Map::Bits1::IsCallableBit::kMask,
                __ Word32BitwiseAnd(
                    bitfield, (Map::Bits1::IsCallableBit::kMask) |
                                  (Map::Bits1::IsUndetectableBit::kMask)));
            break;
          case ObjectIsOp::Kind::kNonCallable:
            check = __ Word32Equal(
                0, __ Word32BitwiseAnd(bitfield,
                                       Map::Bits1::IsCallableBit::kMask));
            GOTO_IF_NOT(check, done, 0);
            // Fallthrough into receiver check.
            V8_FALLTHROUGH;
          case ObjectIsOp::Kind::kReceiver:
            check = JSAnyIsNotPrimitiveHeapObject(input, map);
            break;
          case ObjectIsOp::Kind::kReceiverOrNullOrUndefined: {
#if V8_STATIC_ROOTS_BOOL
            V<Word32> check0 = JSAnyIsNotPrimitiveHeapObject(input, map);
            V<Word32> check1 = __ TaggedEqual(
                input, __ HeapConstant(factory_->undefined_value()));
            V<Word32> check2 =
                __ TaggedEqual(input, __ HeapConstant(factory_->null_value()));
            check =
                __ Word32BitwiseOr(check0, __ Word32BitwiseOr(check1, check2));
#else
            static_assert(LAST_PRIMITIVE_HEAP_OBJECT_TYPE == ODDBALL_TYPE);
            static_assert(LAST_TYPE == LAST_JS_RECEIVER_TYPE);
            // Rule out all primitives except oddballs (true, false, undefined,
            // null).
            V<Word32> instance_type = __ LoadInstanceTypeField(map);
            GOTO_IF_NOT(__ Uint32LessThanOrEqual(ODDBALL_TYPE, instance_type),
                        done, 0);

            // Rule out booleans.
            check = __ Word32Equal(
                0,
                __ TaggedEqual(map, __ HeapConstant(factory_->boolean_map())));
#endif  // V8_STATIC_ROOTS_BOOL
            break;
          }
          case ObjectIsOp::Kind::kUndetectable:
            check = __ Word32Equal(
                Map::Bits1::IsUndetectableBit::kMask,
                __ Word32BitwiseAnd(bitfield,
                                    Map::Bits1::IsUndetectableBit::kMask));
            break;
          default:
            UNREACHABLE();
        }
        GOTO(done, check);

        BIND(done, result);
        return result;
      }
      case ObjectIsOp::Kind::kSmi: {
        // If we statically know that this is a heap object, it cannot be a Smi.
        if (!NeedsHeapObjectCheck(input_assumptions)) {
          return __ Word32Constant(0);
        }
        return __ IsSmi(input);
      }
      case ObjectIsOp::Kind::kNumber: {
        Label<Word32> done(this);

        // Check for Smi if necessary.
        if (NeedsHeapObjectCheck(input_assumptions)) {
          GOTO_IF(__ IsSmi(input), done, 1);
        }

        V<Map> map = __ LoadMapField(input);
        GOTO(done,
             __ TaggedEqual(map, __ HeapConstant(factory_->heap_number_map())));

        BIND(done, result);
        return result;
      }

#if V8_STATIC_ROOTS_BOOL
      case ObjectIsOp::Kind::kString: {
        Label<Word32> done(this);

        // Check for Smi if necessary.
        if (NeedsHeapObjectCheck(input_assumptions)) {
          GOTO_IF(__ IsSmi(input), done, 0);
        }

        V<Map> map = __ LoadMapField(input);
        GOTO(done, __ Uint32LessThanOrEqual(
                       __ TruncateWordPtrToWord32(__ BitcastTaggedToWord(map)),
                       __ Word32Constant(InstanceTypeChecker::kLastStringMap)));

        BIND(done, result);
        return result;
      }
      case ObjectIsOp::Kind::kSymbol: {
        Label<Word32> done(this);

        // Check for Smi if necessary.
        if (NeedsHeapObjectCheck(input_assumptions)) {
          GOTO_IF(__ IsSmi(input), done, 0);
        }

        V<Map> map = __ LoadMapField(input);
        GOTO(done, __ Word32Equal(
                       __ TruncateWordPtrToWord32(__ BitcastTaggedToWord(map)),
                       __ Word32Constant(StaticReadOnlyRoot::kSymbolMap)));

        BIND(done, result);
        return result;
      }
#else
      case ObjectIsOp::Kind::kString:
      case ObjectIsOp::Kind::kSymbol:
#endif
      case ObjectIsOp::Kind::kArrayBufferView: {
        Label<Word32> done(this);

        // Check for Smi if necessary.
        if (NeedsHeapObjectCheck(input_assumptions)) {
          GOTO_IF(__ IsSmi(input), done, 0);
        }

        // Load instance type from map.
        V<Map> map = __ LoadMapField(input);
        V<Word32> instance_type = __ LoadInstanceTypeField(map);

        V<Word32> check;
        switch (kind) {
#if !V8_STATIC_ROOTS_BOOL
          case ObjectIsOp::Kind::kSymbol:
            check = __ Word32Equal(instance_type, SYMBOL_TYPE);
            break;
          case ObjectIsOp::Kind::kString:
            check = __ Uint32LessThan(instance_type, FIRST_NONSTRING_TYPE);
            break;
#endif
          case ObjectIsOp::Kind::kArrayBufferView:
            check = __ Uint32LessThan(
                __ Word32Sub(instance_type, FIRST_JS_ARRAY_BUFFER_VIEW_TYPE),
                LAST_JS_ARRAY_BUFFER_VIEW_TYPE -
                    FIRST_JS_ARRAY_BUFFER_VIEW_TYPE + 1);
            break;
          default:
            UNREACHABLE();
        }
        GOTO(done, check);

        BIND(done, result);
        return result;
      }
      case ObjectIsOp::Kind::kInternalizedString: {
        DCHECK_EQ(input_assumptions, ObjectIsOp::InputAssumptions::kHeapObject);
        // Load instance type from map.
        V<Map> map = __ LoadMapField(input);
        V<Word32> instance_type = __ LoadInstanceTypeField(map);

        return __ Word32Equal(
            __ Word32BitwiseAnd(instance_type,
                                (kIsNotStringMask | kIsNotInternalizedMask)),
            kInternalizedTag);
      }
    }

    UNREACHABLE();
  }

  V<Word32> REDUCE(FloatIs)(OpIndex value, NumericKind kind,
                            FloatRepresentation input_rep) {
    DCHECK_EQ(input_rep, FloatRepresentation::Float64());
    switch (kind) {
      case NumericKind::kFloat64Hole: {
        Label<Word32> done(this);
        // First check whether {value} is a NaN at all...
        GOTO_IF(LIKELY(__ Float64Equal(value, value)), done, 0);
        // ...and only if {value} is a NaN, perform the expensive bit
        // check. See http://crbug.com/v8/8264 for details.
        GOTO(done, __ Word32Equal(__ Float64ExtractHighWord32(value),
                                  kHoleNanUpper32));
        BIND(done, result);
        return result;
      }
      case NumericKind::kFinite: {
        V<Float64> diff = __ Float64Sub(value, value);
        return __ Float64Equal(diff, diff);
      }
      case NumericKind::kInteger: {
        V<Float64> trunc = __ Float64RoundToZero(value);
        V<Float64> diff = __ Float64Sub(value, trunc);
        return __ Float64Equal(diff, 0.0);
      }
      case NumericKind::kSafeInteger: {
        Label<Word32> done(this);
        V<Float64> trunc = __ Float64RoundToZero(value);
        V<Float64> diff = __ Float64Sub(value, trunc);
        GOTO_IF_NOT(__ Float64Equal(diff, 0), done, 0);
        V<Word32> in_range =
            __ Float64LessThanOrEqual(__ Float64Abs(trunc), kMaxSafeInteger);
        GOTO(done, in_range);

        BIND(done, result);
        return result;
      }
      case NumericKind::kMinusZero: {
        if (Is64()) {
          V<Word64> value64 = __ BitcastFloat64ToWord64(value);
          return __ Word64Equal(value64, kMinusZeroBits);
        } else {
          Label<Word32> done(this);
          V<Word32> value_lo = __ Float64ExtractLowWord32(value);
          GOTO_IF_NOT(__ Word32Equal(value_lo, kMinusZeroLoBits), done, 0);
          V<Word32> value_hi = __ Float64ExtractHighWord32(value);
          GOTO(done, __ Word32Equal(value_hi, kMinusZeroHiBits));

          BIND(done, result);
          return result;
        }
      }
      case NumericKind::kNaN: {
        V<Word32> diff = __ Float64Equal(value, value);
        return __ Word32Equal(diff, 0);
      }
    }

    UNREACHABLE();
  }

  V<Word32> REDUCE(ObjectIsNumericValue)(V<Object> input, NumericKind kind,
                                         FloatRepresentation input_rep) {
    DCHECK_EQ(input_rep, FloatRepresentation::Float64());
    Label<Word32> done(this);

    switch (kind) {
      case NumericKind::kFinite:
      case NumericKind::kInteger:
      case NumericKind::kSafeInteger:
        GOTO_IF(__ IsSmi(input), done, 1);
        break;
      case NumericKind::kMinusZero:
      case NumericKind::kNaN:
        GOTO_IF(__ IsSmi(input), done, 0);
        break;
      case NumericKind::kFloat64Hole:
        // ObjectIsFloat64Hole is not used, but can be implemented when needed.
        UNREACHABLE();
    }

    V<Map> map = __ LoadMapField(input);
    GOTO_IF_NOT(
        __ TaggedEqual(map, __ HeapConstant(factory_->heap_number_map())), done,
        0);

    V<Float64> value = __ template LoadField<Float64>(
        input, AccessBuilder::ForHeapNumberValue());
    GOTO(done, __ FloatIs(value, kind, input_rep));

    BIND(done, result);
    return result;
  }

  V<Object> REDUCE(Convert)(V<Object> input, ConvertOp::Kind from,
                            ConvertOp::Kind to) {
    switch (to) {
      case ConvertOp::Kind::kNumber: {
        if (from == ConvertOp::Kind::kPlainPrimitive) {
          return __ CallBuiltin_PlainPrimitiveToNumber(
              isolate_, V<PlainPrimitive>::Cast(input));
        } else {
          DCHECK_EQ(from, ConvertOp::Kind::kString);
          return __ CallBuiltin_StringToNumber(isolate_,
                                               V<String>::Cast(input));
        }
      }
      case ConvertOp::Kind::kBoolean: {
        DCHECK_EQ(from, ConvertOp::Kind::kObject);
        return __ CallBuiltin_ToBoolean(isolate_, input);
      }
      case ConvertOp::Kind::kString: {
        DCHECK_EQ(from, ConvertOp::Kind::kNumber);
        return __ CallBuiltin_NumberToString(isolate_, V<Number>::Cast(input));
      }
      case ConvertOp::Kind::kSmi: {
        DCHECK_EQ(from, ConvertOp::Kind::kNumberOrOddball);
        Label<Smi> done(this);
        GOTO_IF(LIKELY(__ ObjectIsSmi(input)), done, V<Smi>::Cast(input));

        V<Float64> value = __ template LoadField<Float64>(
            input, AccessBuilder::ForHeapNumberOrOddballValue());
        GOTO(done, __ TagSmi(__ ReversibleFloat64ToInt32(value)));

        BIND(done, result);
        return result;
      }
      default:
        UNREACHABLE();
    }
  }

  V<Object> REDUCE(ConvertUntaggedToJSPrimitive)(
      OpIndex input, ConvertUntaggedToJSPrimitiveOp::JSPrimitiveKind kind,
      RegisterRepresentation input_rep,
      ConvertUntaggedToJSPrimitiveOp::InputInterpretation input_interpretation,
      CheckForMinusZeroMode minus_zero_mode) {
    switch (kind) {
      case ConvertUntaggedToJSPrimitiveOp::JSPrimitiveKind::kBigInt: {
        DCHECK(Is64());
        DCHECK_EQ(input_rep, RegisterRepresentation::Word64());
        Label<Tagged> done(this);

        // BigInts with value 0 must be of size 0 (canonical form).
        GOTO_IF(__ Word64Equal(input, int64_t{0}), done,
                AllocateBigInt(OpIndex::Invalid(), OpIndex::Invalid()));

        // The GOTO_IF above could have been changed to an unconditional GOTO,
        // in which case we are now in unreachable code, so we can skip the
        // following step and return.
        if (input_interpretation ==
            ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kSigned) {
          // Shift sign bit into BigInt's sign bit position.
          V<Word32> bitfield = __ Word32BitwiseOr(
              BigInt::LengthBits::encode(1),
              __ TruncateWord64ToWord32(__ Word64ShiftRightLogical(
                  input, static_cast<int32_t>(63 - BigInt::SignBits::kShift))));

          // We use (value XOR (value >> 63)) - (value >> 63) to compute the
          // absolute value, in a branchless fashion.
          V<Word64> sign_mask =
              __ Word64ShiftRightArithmetic(input, int32_t{63});
          V<Word64> absolute_value =
              __ Word64Sub(__ Word64BitwiseXor(input, sign_mask), sign_mask);
          GOTO(done, AllocateBigInt(bitfield, absolute_value));
        } else {
          DCHECK_EQ(
              input_interpretation,
              ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kUnsigned);
          const auto bitfield = BigInt::LengthBits::encode(1);
          GOTO(done, AllocateBigInt(__ Word32Constant(bitfield), input));
        }

        BIND(done, result);
        return result;
      }
      case ConvertUntaggedToJSPrimitiveOp::JSPrimitiveKind::kNumber: {
        if (input_rep == RegisterRepresentation::Word32()) {
          switch (input_interpretation) {
            case ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kSigned: {
              if (SmiValuesAre32Bits()) {
                return __ TagSmi(input);
              }
              DCHECK(SmiValuesAre31Bits());

              Label<Tagged> done(this);
              Label<> overflow(this);

              TagSmiOrOverflow(input, &overflow, &done);

              if (BIND(overflow)) {
                GOTO(done, AllocateHeapNumberWithValue(
                               __ ChangeInt32ToFloat64(input)));
              }

              BIND(done, result);
              return result;
            }
            case ConvertUntaggedToJSPrimitiveOp::InputInterpretation::
                kUnsigned: {
              Label<Tagged> done(this);

              GOTO_IF(__ Uint32LessThanOrEqual(input, Smi::kMaxValue), done,
                      __ TagSmi(input));
              GOTO(done, AllocateHeapNumberWithValue(
                             __ ChangeUint32ToFloat64(input)));

              BIND(done, result);
              return result;
            }
            case ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kCharCode:
            case ConvertUntaggedToJSPrimitiveOp::InputInterpretation::
                kCodePoint:
              UNREACHABLE();
          }
        } else if (input_rep == RegisterRepresentation::Word64()) {
          switch (input_interpretation) {
            case ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kSigned: {
              Label<Tagged> done(this);
              Label<> outside_smi_range(this);

              V<Word32> v32 = __ TruncateWord64ToWord32(input);
              V<Word64> v64 = __ ChangeInt32ToInt64(v32);
              GOTO_IF_NOT(__ Word64Equal(v64, input), outside_smi_range);

              if constexpr (SmiValuesAre32Bits()) {
                GOTO(done, __ TagSmi(input));
              } else {
                TagSmiOrOverflow(v32, &outside_smi_range, &done);
              }

              if (BIND(outside_smi_range)) {
                GOTO(done, AllocateHeapNumberWithValue(
                               __ ChangeInt64ToFloat64(input)));
              }

              BIND(done, result);
              return result;
            }
            case ConvertUntaggedToJSPrimitiveOp::InputInterpretation::
                kUnsigned: {
              Label<Tagged> done(this);

              GOTO_IF(__ Uint64LessThanOrEqual(input, Smi::kMaxValue), done,
                      __ TagSmi(__ TruncateWord64ToWord32(input)));
              GOTO(done,
                   AllocateHeapNumberWithValue(__ ChangeInt64ToFloat64(input)));

              BIND(done, result);
              return result;
            }
            case ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kCharCode:
            case ConvertUntaggedToJSPrimitiveOp::InputInterpretation::
                kCodePoint:
              UNREACHABLE();
          }
        } else {
          DCHECK_EQ(input_rep, RegisterRepresentation::Float64());
          Label<Tagged> done(this);
          Label<> outside_smi_range(this);

          V<Word32> v32 = __ TruncateFloat64ToInt32OverflowUndefined(input);
          GOTO_IF_NOT(__ Float64Equal(input, __ ChangeInt32ToFloat64(v32)),
                      outside_smi_range);

          if (minus_zero_mode == CheckForMinusZeroMode::kCheckForMinusZero) {
            // In case of 0, we need to check the high bits for the IEEE -0
            // pattern.
            IF(__ Word32Equal(v32, 0)) {
              GOTO_IF(__ Int32LessThan(__ Float64ExtractHighWord32(input), 0),
                      outside_smi_range);
            }
            END_IF
          }

          if constexpr (SmiValuesAre32Bits()) {
            GOTO(done, __ TagSmi(v32));
          } else {
            TagSmiOrOverflow(v32, &outside_smi_range, &done);
          }

          if (BIND(outside_smi_range)) {
            GOTO(done, AllocateHeapNumberWithValue(input));
          }

          BIND(done, result);
          return result;
        }
        UNREACHABLE();
        break;
      }
      case ConvertUntaggedToJSPrimitiveOp::JSPrimitiveKind::kHeapNumber: {
        DCHECK_EQ(input_rep, RegisterRepresentation::Float64());
        DCHECK_EQ(input_interpretation,
                  ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kSigned);
        return AllocateHeapNumberWithValue(input);
      }
      case ConvertUntaggedToJSPrimitiveOp::JSPrimitiveKind::kSmi: {
        DCHECK_EQ(input_rep, RegisterRepresentation::Word32());
        DCHECK_EQ(input_interpretation,
                  ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kSigned);
        return __ TagSmi(input);
      }
      case ConvertUntaggedToJSPrimitiveOp::JSPrimitiveKind::kBoolean: {
        DCHECK_EQ(input_rep, RegisterRepresentation::Word32());
        DCHECK_EQ(input_interpretation,
                  ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kSigned);
        Label<Tagged> done(this);

        IF(input) { GOTO(done, __ HeapConstant(factory_->true_value())); }
        ELSE { GOTO(done, __ HeapConstant(factory_->false_value())); }
        END_IF

        BIND(done, result);
        return result;
      }
      case ConvertUntaggedToJSPrimitiveOp::JSPrimitiveKind::kString: {
        Label<Word32> single_code(this);
        Label<Tagged> done(this);

        if (input_interpretation ==
            ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kCharCode) {
          GOTO(single_code, __ Word32BitwiseAnd(input, 0xFFFF));
        } else {
          DCHECK_EQ(
              input_interpretation,
              ConvertUntaggedToJSPrimitiveOp::InputInterpretation::kCodePoint);
          // Check if the input is a single code unit.
          GOTO_IF(LIKELY(__ Uint32LessThanOrEqual(input, 0xFFFF)), single_code,
                  input);

          // Generate surrogate pair string.

          // Convert UTF32 to UTF16 code units and store as a 32 bit word.
          V<Word32> lead_offset = __ Word32Constant(0xD800 - (0x10000 >> 10));

          // lead = (codepoint >> 10) + LEAD_OFFSET
          V<Word32> lead =
              __ Word32Add(__ Word32ShiftRightLogical(input, 10), lead_offset);

          // trail = (codepoint & 0x3FF) + 0xDC00
          V<Word32> trail =
              __ Word32Add(__ Word32BitwiseAnd(input, 0x3FF), 0xDC00);

          // codepoint = (trail << 16) | lead
#if V8_TARGET_BIG_ENDIAN
          V<Word32> code =
              __ Word32BitwiseOr(__ Word32ShiftLeft(lead, 16), trail);
#else
          V<Word32> code =
              __ Word32BitwiseOr(__ Word32ShiftLeft(trail, 16), lead);
#endif

          // Allocate a new SeqTwoByteString for {code}.
          auto string = __ template Allocate<String>(
              __ IntPtrConstant(SeqTwoByteString::SizeFor(2)),
              AllocationType::kYoung);
          // Set padding to 0.
          __ Initialize(string, __ IntPtrConstant(0),
                        MemoryRepresentation::TaggedSigned(),
                        WriteBarrierKind::kNoWriteBarrier,
                        SeqTwoByteString::SizeFor(2) - kObjectAlignment);
          __ InitializeField(
              string, AccessBuilder::ForMap(),
              __ HeapConstant(factory_->seq_two_byte_string_map()));
          __ InitializeField(string, AccessBuilder::ForNameRawHashField(),
                             __ Word32Constant(Name::kEmptyHashField));
          __ InitializeField(string, AccessBuilder::ForStringLength(),
                             __ Word32Constant(2));
          // Write the code as a single 32-bit value by adapting the elements
          // access to SeqTwoByteString characters.
          ElementAccess char_access =
              AccessBuilder::ForSeqTwoByteStringCharacter();
          char_access.machine_type = MachineType::Uint32();
          __ InitializeNonArrayBufferElement(string, char_access,
                                             __ IntPtrConstant(0), code);
          GOTO(done, __ FinishInitialization(std::move(string)));
        }

        if (BIND(single_code, code)) {
          // Check if the {code} is a one byte character.
          IF(LIKELY(
              __ Uint32LessThanOrEqual(code, String::kMaxOneByteCharCode))) {
            // Load the isolate wide single character string table.
            V<Tagged> table =
                __ HeapConstant(factory_->single_character_string_table());

            // Compute the {table} index for {code}.
            V<WordPtr> index = __ ChangeUint32ToUintPtr(code);

            // Load the string for the {code} from the single character string
            // table.
            OpIndex entry = __ LoadNonArrayBufferElement(
                table, AccessBuilder::ForFixedArrayElement(), index);

            // Use the {entry} from the {table}.
            GOTO(done, entry);
          }
          ELSE {
            // Allocate a new SeqTwoBytesString for {code}.
            auto string = __ template Allocate<String>(
                __ IntPtrConstant(SeqTwoByteString::SizeFor(1)),
                AllocationType::kYoung);

            // Set padding to 0.
            __ Initialize(string, __ IntPtrConstant(0),
                          MemoryRepresentation::TaggedSigned(),
                          WriteBarrierKind::kNoWriteBarrier,
                          SeqTwoByteString::SizeFor(1) - kObjectAlignment);
            __ InitializeField(
                string, AccessBuilder::ForMap(),
                __ HeapConstant(factory_->seq_two_byte_string_map()));
            __ InitializeField(string, AccessBuilder::ForNameRawHashField(),
                               __ Word32Constant(Name::kEmptyHashField));
            __ InitializeField(string, AccessBuilder::ForStringLength(),
                               __ Word32Constant(1));
            __ InitializeNonArrayBufferElement(
                string, AccessBuilder::ForSeqTwoByteStringCharacter(),
                __ IntPtrConstant(0), code);
            GOTO(done, __ FinishInitialization(std::move(string)));
          }
          END_IF
        }

        BIND(done, result);
        return result;
      }
    }

    UNREACHABLE();
  }

  OpIndex REDUCE(ConvertUntaggedToJSPrimitiveOrDeopt)(
      OpIndex input, OpIndex frame_state,
      ConvertUntaggedToJSPrimitiveOrDeoptOp::JSPrimitiveKind kind,
      RegisterRepresentation input_rep,
      ConvertUntaggedToJSPrimitiveOrDeoptOp::InputInterpretation
          input_interpretation,
      const FeedbackSource& feedback) {
    DCHECK_EQ(kind,
              ConvertUntaggedToJSPrimitiveOrDeoptOp::JSPrimitiveKind::kSmi);
    if (input_rep == RegisterRepresentation::Word32()) {
      if (input_interpretation ==
          ConvertUntaggedToJSPrimitiveOrDeoptOp::InputInterpretation::kSigned) {
        if constexpr (SmiValuesAre32Bits()) {
          return __ TagSmi(input);
        } else {
          OpIndex test = __ Int32AddCheckOverflow(input, input);
          __ DeoptimizeIf(__ template Projection<Word32>(test, 1), frame_state,
                          DeoptimizeReason::kLostPrecision, feedback);
          return __ BitcastWord32ToTagged(
              __ template Projection<Word32>(test, 0));
        }
      } else {
        DCHECK_EQ(input_interpretation, ConvertUntaggedToJSPrimitiveOrDeoptOp::
                                            InputInterpretation::kUnsigned);
        V<Word32> check = __ Uint32LessThanOrEqual(input, Smi::kMaxValue);
        __ DeoptimizeIfNot(check, frame_state, DeoptimizeReason::kLostPrecision,
                           feedback);
        return __ TagSmi(input);
      }
    } else {
      DCHECK_EQ(input_rep, RegisterRepresentation::Word64());
      if (input_interpretation ==
          ConvertUntaggedToJSPrimitiveOrDeoptOp::InputInterpretation::kSigned) {
        V<Word32> i32 = __ TruncateWord64ToWord32(input);
        V<Word32> check = __ Word64Equal(__ ChangeInt32ToInt64(i32), input);
        __ DeoptimizeIfNot(check, frame_state, DeoptimizeReason::kLostPrecision,
                           feedback);
        if constexpr (SmiValuesAre32Bits()) {
          return __ TagSmi(input);
        } else {
          OpIndex test = __ Int32AddCheckOverflow(i32, i32);
          __ DeoptimizeIf(__ template Projection<Word32>(test, 1), frame_state,
                          DeoptimizeReason::kLostPrecision, feedback);
          return __ BitcastWord32ToTagged(
              __ template Projection<Word32>(test, 0));
        }
      } else {
        DCHECK_EQ(input_interpretation, ConvertUntaggedToJSPrimitiveOrDeoptOp::
                                            InputInterpretation::kUnsigned);
        V<Word32> check = __ Uint64LessThanOrEqual(
            input, static_cast<uint64_t>(Smi::kMaxValue));
        __ DeoptimizeIfNot(check, frame_state, DeoptimizeReason::kLostPrecision,
                           feedback);
        return __ TagSmi(input);
      }
    }

    UNREACHABLE();
  }

  OpIndex REDUCE(ConvertJSPrimitiveToUntagged)(
      V<Object> object, ConvertJSPrimitiveToUntaggedOp::UntaggedKind kind,
      ConvertJSPrimitiveToUntaggedOp::InputAssumptions input_assumptions) {
    switch (kind) {
      case ConvertJSPrimitiveToUntaggedOp::UntaggedKind::kInt32:
        if (input_assumptions ==
            ConvertJSPrimitiveToUntaggedOp::InputAssumptions::kSmi) {
          return __ UntagSmi(object);
        } else if (input_assumptions ==
                   ConvertJSPrimitiveToUntaggedOp::InputAssumptions::
                       kNumberOrOddball) {
          Label<Word32> done(this);

          IF (LIKELY(__ ObjectIsSmi(object))) {
            GOTO(done, __ UntagSmi(object));
          }
          ELSE {
            V<Float64> value = __ template LoadField<Float64>(
                object, AccessBuilder::ForHeapNumberOrOddballValue());
            GOTO(done, __ ReversibleFloat64ToInt32(value));
          }
          END_IF

          BIND(done, result);
          return result;
        } else {
          DCHECK_EQ(input_assumptions, ConvertJSPrimitiveToUntaggedOp::
                                           InputAssumptions::kPlainPrimitive);
          Label<Word32> done(this);
          GOTO_IF(LIKELY(__ ObjectIsSmi(object)), done, __ UntagSmi(object));
          V<Number> number =
              __ ConvertPlainPrimitiveToNumber(V<PlainPrimitive>::Cast(object));
          GOTO_IF(__ ObjectIsSmi(number), done, __ UntagSmi(number));
          V<Float64> f64 = __ template LoadField<Float64>(
              V<HeapNumber>::Cast(number), AccessBuilder::ForHeapNumberValue());
          GOTO(done, __ JSTruncateFloat64ToWord32(f64));
          BIND(done, result);
          return result;
        }
        UNREACHABLE();
      case ConvertJSPrimitiveToUntaggedOp::UntaggedKind::kInt64:
        if (input_assumptions ==
            ConvertJSPrimitiveToUntaggedOp::InputAssumptions::kSmi) {
          return __ ChangeInt32ToInt64(__ UntagSmi(object));
        } else {
          DCHECK_EQ(input_assumptions, ConvertJSPrimitiveToUntaggedOp::
                                           InputAssumptions::kNumberOrOddball);
          Label<Word64> done(this);

          IF (LIKELY(__ ObjectIsSmi(object))) {
            GOTO(done, __ ChangeInt32ToInt64(__ UntagSmi(object)));
          }
          ELSE {
            V<Float64> value = __ template LoadField<Float64>(
                object, AccessBuilder::ForHeapNumberOrOddballValue());
            GOTO(done, __ ReversibleFloat64ToInt64(value));
          }
          END_IF

          BIND(done, result);
          return result;
        }
        UNREACHABLE();
      case ConvertJSPrimitiveToUntaggedOp::UntaggedKind::kUint32: {
        DCHECK_EQ(
            input_assumptions,
            ConvertJSPrimitiveToUntaggedOp::InputAssumptions::kNumberOrOddball);
        Label<Word32> done(this);

        IF (LIKELY(__ ObjectIsSmi(object))) {
          GOTO(done, __ UntagSmi(object));
        }
        ELSE {
          V<Float64> value = __ template LoadField<Float64>(
              object, AccessBuilder::ForHeapNumberOrOddballValue());
          GOTO(done, __ ReversibleFloat64ToUint32(value));
        }
        END_IF

        BIND(done, result);
        return result;
      }
      case ConvertJSPrimitiveToUntaggedOp::UntaggedKind::kBit:
        DCHECK_EQ(input_assumptions,
                  ConvertJSPrimitiveToUntaggedOp::InputAssumptions::kBoolean);
        return __ TaggedEqual(object, __ HeapConstant(factory_->true_value()));
      case ConvertJSPrimitiveToUntaggedOp::UntaggedKind::kFloat64: {
        if (input_assumptions == ConvertJSPrimitiveToUntaggedOp::
                                     InputAssumptions::kNumberOrOddball) {
          Label<Float64> done(this);

          IF (LIKELY(__ ObjectIsSmi(object))) {
            GOTO(done, __ ChangeInt32ToFloat64(__ UntagSmi(object)));
          }
          ELSE {
            V<Float64> value = __ template LoadField<Float64>(
                object, AccessBuilder::ForHeapNumberOrOddballValue());
            GOTO(done, value);
          }
          END_IF

          BIND(done, result);
          return result;
        } else {
          DCHECK_EQ(input_assumptions, ConvertJSPrimitiveToUntaggedOp::
                                           InputAssumptions::kPlainPrimitive);
          Label<Float64> done(this);
          GOTO_IF(LIKELY(__ ObjectIsSmi(object)), done,
                  __ ChangeInt32ToFloat64(__ UntagSmi(object)));
          V<Number> number =
              __ ConvertPlainPrimitiveToNumber(V<PlainPrimitive>::Cast(object));
          GOTO_IF(__ ObjectIsSmi(number), done,
                  __ ChangeInt32ToFloat64(__ UntagSmi(number)));
          V<Float64> f64 = __ template LoadField<Float64>(
              V<HeapNumber>::Cast(number), AccessBuilder::ForHeapNumberValue());
          GOTO(done, f64);
          BIND(done, result);
          return result;
        }
      }
    }
    UNREACHABLE();
  }

  OpIndex REDUCE(ConvertJSPrimitiveToUntaggedOrDeopt)(
      V<Tagged> object, OpIndex frame_state,
      ConvertJSPrimitiveToUntaggedOrDeoptOp::JSPrimitiveKind from_kind,
      ConvertJSPrimitiveToUntaggedOrDeoptOp::UntaggedKind to_kind,
      CheckForMinusZeroMode minus_zero_mode, const FeedbackSource& feedback) {
    switch (to_kind) {
      case ConvertJSPrimitiveToUntaggedOrDeoptOp::UntaggedKind::kInt32: {
        if (from_kind ==
            ConvertJSPrimitiveToUntaggedOrDeoptOp::JSPrimitiveKind::kSmi) {
          __ DeoptimizeIfNot(__ ObjectIsSmi(object), frame_state,
                             DeoptimizeReason::kNotASmi, feedback);
          return __ UntagSmi(object);
        } else {
          DCHECK_EQ(
              from_kind,
              ConvertJSPrimitiveToUntaggedOrDeoptOp::JSPrimitiveKind::kNumber);
          Label<Word32> done(this);

          IF(LIKELY(__ ObjectIsSmi(object))) {
            GOTO(done, __ UntagSmi(object));
          }
          ELSE {
            V<Map> map = __ LoadMapField(object);
            __ DeoptimizeIfNot(
                __ TaggedEqual(map,
                               __ HeapConstant(factory_->heap_number_map())),
                frame_state, DeoptimizeReason::kNotAHeapNumber, feedback);
            V<Float64> heap_number_value = __ template LoadField<Float64>(
                object, AccessBuilder::ForHeapNumberValue());

            GOTO(done,
                 __ ChangeFloat64ToInt32OrDeopt(heap_number_value, frame_state,
                                                minus_zero_mode, feedback));
          }
          END_IF

          BIND(done, result);
          return result;
        }
      }
      case ConvertJSPrimitiveToUntaggedOrDeoptOp::UntaggedKind::kInt64: {
        DCHECK_EQ(
            from_kind,
            ConvertJSPrimitiveToUntaggedOrDeoptOp::JSPrimitiveKind::kNumber);
        Label<Word64> done(this);

        IF(LIKELY(__ ObjectIsSmi(object))) {
          GOTO(done, __ ChangeInt32ToInt64(__ UntagSmi(object)));
        }
        ELSE {
          V<Map> map = __ LoadMapField(object);
          __ DeoptimizeIfNot(
              __ TaggedEqual(map, __ HeapConstant(factory_->heap_number_map())),
              frame_state, DeoptimizeReason::kNotAHeapNumber, feedback);
          V<Float64> heap_number_value = __ template LoadField<Float64>(
              object, AccessBuilder::ForHeapNumberValue());
          GOTO(done,
               __ ChangeFloat64ToInt64OrDeopt(heap_number_value, frame_state,
                                              minus_zero_mode, feedback));
        }
        END_IF

        BIND(done, result);
        return result;
      }
      case ConvertJSPrimitiveToUntaggedOrDeoptOp::UntaggedKind::kFloat64: {
        Label<Float64> done(this);

        // In the Smi case, just convert to int32 and then float64.
        // Otherwise, check heap numberness and load the number.
        IF(__ ObjectIsSmi(object)) {
          GOTO(done, __ ChangeInt32ToFloat64(__ UntagSmi(object)));
        }
        ELSE {
          GOTO(done, ConvertHeapObjectToFloat64OrDeopt(object, frame_state,
                                                       from_kind, feedback));
        }
        END_IF

        BIND(done, result);
        return result;
      }
      case ConvertJSPrimitiveToUntaggedOrDeoptOp::UntaggedKind::kArrayIndex: {
        DCHECK_EQ(from_kind, ConvertJSPrimitiveToUntaggedOrDeoptOp::
                                 JSPrimitiveKind::kNumberOrString);
        Label<WordPtr> done(this);

        IF(LIKELY(__ ObjectIsSmi(object))) {
          // In the Smi case, just convert to intptr_t.
          GOTO(done, __ ChangeInt32ToIntPtr(__ UntagSmi(object)));
        }
        ELSE {
          V<Map> map = __ LoadMapField(object);
          IF(LIKELY(__ TaggedEqual(
              map, __ HeapConstant(factory_->heap_number_map())))) {
            V<Float64> heap_number_value = __ template LoadField<Float64>(
                object, AccessBuilder::ForHeapNumberValue());
            // Perform Turbofan's "CheckedFloat64ToIndex"
            {
              if constexpr (Is64()) {
                V<Word64> i64 = __ TruncateFloat64ToInt64OverflowUndefined(
                    heap_number_value);
                // The TruncateKind above means there will be a precision loss
                // in case INT64_MAX input is passed, but that precision loss
                // would not be detected and would not lead to a deoptimization
                // from the first check. But in this case, we'll deopt anyway
                // because of the following checks.
                __ DeoptimizeIfNot(__ Float64Equal(__ ChangeInt64ToFloat64(i64),
                                                   heap_number_value),
                                   frame_state,
                                   DeoptimizeReason::kLostPrecisionOrNaN,
                                   feedback);
                __ DeoptimizeIfNot(
                    __ IntPtrLessThan(i64, kMaxSafeIntegerUint64), frame_state,
                    DeoptimizeReason::kNotAnArrayIndex, feedback);
                __ DeoptimizeIfNot(
                    __ IntPtrLessThan(-kMaxSafeIntegerUint64, i64), frame_state,
                    DeoptimizeReason::kNotAnArrayIndex, feedback);
                GOTO(done, i64);
              } else {
                V<Word32> i32 = __ TruncateFloat64ToInt32OverflowUndefined(
                    heap_number_value);
                __ DeoptimizeIfNot(__ Float64Equal(__ ChangeInt32ToFloat64(i32),
                                                   heap_number_value),
                                   frame_state,
                                   DeoptimizeReason::kLostPrecisionOrNaN,
                                   feedback);
                GOTO(done, i32);
              }
            }
          }
          ELSE {
#if V8_STATIC_ROOTS_BOOL
            V<Word32> is_string_map = __ Uint32LessThanOrEqual(
                __ TruncateWordPtrToWord32(__ BitcastTaggedToWord(map)),
                __ Word32Constant(InstanceTypeChecker::kLastStringMap));
#else
            V<Word32> instance_type = __ LoadInstanceTypeField(map);
            V<Word32> is_string_map =
                __ Uint32LessThan(instance_type, FIRST_NONSTRING_TYPE);
#endif
            __ DeoptimizeIfNot(is_string_map, frame_state,
                               DeoptimizeReason::kNotAString, feedback);

            // TODO(nicohartmann@): We might introduce a Turboshaft way for
            // constructing call descriptors.
            MachineSignature::Builder builder(__ graph_zone(), 1, 1);
            builder.AddReturn(MachineType::Int32());
            builder.AddParam(MachineType::TaggedPointer());
            auto desc = Linkage::GetSimplifiedCDescriptor(__ graph_zone(),
                                                          builder.Build());
            auto ts_desc =
                TSCallDescriptor::Create(desc, CanThrow::kNo, __ graph_zone());
            OpIndex callee = __ ExternalConstant(
                ExternalReference::string_to_array_index_function());
            // NOTE: String::ToArrayIndex() currently returns int32_t.
            V<WordPtr> index =
                __ ChangeInt32ToIntPtr(__ Call(callee, {object}, ts_desc));
            __ DeoptimizeIf(__ WordPtrEqual(index, -1), frame_state,
                            DeoptimizeReason::kNotAnArrayIndex, feedback);
            GOTO(done, index);
          }
          END_IF
        }
        END_IF

        BIND(done, result);
        return result;
      }
    }
    UNREACHABLE();
  }

  OpIndex REDUCE(TruncateJSPrimitiveToUntagged)(
      V<Object> object, TruncateJSPrimitiveToUntaggedOp::UntaggedKind kind,
      TruncateJSPrimitiveToUntaggedOp::InputAssumptions input_assumptions) {
    switch (kind) {
      case TruncateJSPrimitiveToUntaggedOp::UntaggedKind::kInt32: {
        DCHECK_EQ(input_assumptions, TruncateJSPrimitiveToUntaggedOp::
                                         InputAssumptions::kNumberOrOddball);
        Label<Word32> done(this);

        IF (LIKELY(__ ObjectIsSmi(object))) {
          GOTO(done, __ UntagSmi(object));
        }
        ELSE {
          V<Float64> number_value = __ template LoadField<Float64>(
              object, AccessBuilder::ForHeapNumberOrOddballValue());
          GOTO(done, __ JSTruncateFloat64ToWord32(number_value));
        }
        END_IF

        BIND(done, result);
        return result;
      }
      case TruncateJSPrimitiveToUntaggedOp::UntaggedKind::kInt64: {
        DCHECK_EQ(input_assumptions,
                  TruncateJSPrimitiveToUntaggedOp::InputAssumptions::kBigInt);
        DCHECK(Is64());
        Label<Word64> done(this);

        V<Word32> bitfield = __ template LoadField<Word32>(
            object, AccessBuilder::ForBigIntBitfield());
        IF(__ Word32Equal(bitfield, 0)) { GOTO(done, 0); }
        ELSE {
          V<Word64> lsd = __ template LoadField<Word64>(
              object, AccessBuilder::ForBigIntLeastSignificantDigit64());
          V<Word32> sign =
              __ Word32BitwiseAnd(bitfield, BigInt::SignBits::kMask);
          IF(__ Word32Equal(sign, 1)) { GOTO(done, __ Word64Sub(0, lsd)); }
          END_IF
          GOTO(done, lsd);
        }
        END_IF

        BIND(done, result);
        return result;
      }
      case TruncateJSPrimitiveToUntaggedOp::UntaggedKind::kBit: {
        Label<Word32> done(this);

        if (input_assumptions ==
            TruncateJSPrimitiveToUntaggedOp::InputAssumptions::kObject) {
          // Perform Smi check.
          IF (UNLIKELY(__ ObjectIsSmi(object))) {
            GOTO(done, __ Word32Equal(__ TaggedEqual(object, __ TagSmi(0)), 0));
          }
          END_IF
          // Otherwise fall through into HeapObject case.
        } else {
          DCHECK_EQ(
              input_assumptions,
              TruncateJSPrimitiveToUntaggedOp::InputAssumptions::kHeapObject);
        }

#if V8_STATIC_ROOTS_BOOL
        // Check if {object} is a falsey root or the true value.
        // Undefined is the first root, so it's the smallest possible pointer
        // value, which means we don't have to subtract it for the range check.
        ReadOnlyRoots roots(isolate_);
        static_assert(StaticReadOnlyRoot::kUndefinedValue + sizeof(Undefined) ==
                      StaticReadOnlyRoot::kNullValue);
        static_assert(StaticReadOnlyRoot::kNullValue + sizeof(Null) ==
                      StaticReadOnlyRoot::kempty_string);
        static_assert(StaticReadOnlyRoot::kempty_string +
                          SeqOneByteString::SizeFor(0) ==
                      StaticReadOnlyRoot::kFalseValue);
        static_assert(StaticReadOnlyRoot::kFalseValue + sizeof(False) ==
                      StaticReadOnlyRoot::kTrueValue);
        V<Word32> object_as_word32 =
            __ TruncateWordPtrToWord32(__ BitcastTaggedToWord(object));
        V<Word32> true_as_word32 =
            __ Word32Constant(StaticReadOnlyRoot::kTrueValue);
        GOTO_IF(__ Uint32LessThan(object_as_word32, true_as_word32), done, 0);
        GOTO_IF(__ Word32Equal(object_as_word32, true_as_word32), done, 1);
#else
        // Check if {object} is false.
        GOTO_IF(
            __ TaggedEqual(object, __ HeapConstant(factory_->false_value())),
            done, 0);

        // Check if {object} is true.
        GOTO_IF(__ TaggedEqual(object, __ HeapConstant(factory_->true_value())),
                done, 1);

        // Check if {object} is the empty string.
        GOTO_IF(
            __ TaggedEqual(object, __ HeapConstant(factory_->empty_string())),
            done, 0);

        // Only check null and undefined if we're not going to check the
        // undetectable bit.
        if (DependOnNoUndetectableObjectsProtector()) {
          // Check if {object} is the null value.
          GOTO_IF(
              __ TaggedEqual(object, __ HeapConstant(factory_->null_value())),
              done, 0);

          // Check if {object} is the undefined value.
          GOTO_IF(__ TaggedEqual(object,
                                 __ HeapConstant(factory_->undefined_value())),
                  done, 0);
        }
#endif

        // Load the map of {object}.
        V<Map> map = __ LoadMapField(object);

        if (!DependOnNoUndetectableObjectsProtector()) {
          // Check if the {object} is undetectable and immediately return false.
          V<Word32> bitfield = __ template LoadField<Word32>(
              map, AccessBuilder::ForMapBitField());
          GOTO_IF(__ Word32BitwiseAnd(bitfield,
                                      Map::Bits1::IsUndetectableBit::kMask),
                  done, 0);
        }

        // Check if {object} is a HeapNumber.
        IF(UNLIKELY(__ TaggedEqual(
            map, __ HeapConstant(factory_->heap_number_map())))) {
          // For HeapNumber {object}, just check that its value is not 0.0, -0.0
          // or NaN.
          V<Float64> number_value = __ template LoadField<Float64>(
              object, AccessBuilder::ForHeapNumberValue());
          GOTO(done, __ Float64LessThan(0.0, __ Float64Abs(number_value)));
        }
        END_IF

        // Check if {object} is a BigInt.
        IF(UNLIKELY(
            __ TaggedEqual(map, __ HeapConstant(factory_->bigint_map())))) {
          V<Word32> bitfield = __ template LoadField<Word32>(
              object, AccessBuilder::ForBigIntBitfield());
          GOTO(done, IsNonZero(__ Word32BitwiseAnd(bitfield,
                                                   BigInt::LengthBits::kMask)));
        }
        END_IF

        // All other values that reach here are true.
        GOTO(done, 1);

        BIND(done, result);
        return result;
      }
    }
    UNREACHABLE();
  }

  OpIndex REDUCE(TruncateJSPrimitiveToUntaggedOrDeopt)(
      V<Object> input, OpIndex frame_state,
      TruncateJSPrimitiveToUntaggedOrDeoptOp::UntaggedKind kind,
      TruncateJSPrimitiveToUntaggedOrDeoptOp::InputRequirement
          input_requirement,
      const FeedbackSource& feedback) {
    DCHECK_EQ(kind,
              TruncateJSPrimitiveToUntaggedOrDeoptOp::UntaggedKind::kInt32);
    Label<Word32> done(this);
    // In the Smi case, just convert to int32.
    GOTO_IF(LIKELY(__ ObjectIsSmi(input)), done, __ UntagSmi(input));

    // Otherwise, check that it's a heap number or oddball and truncate the
    // value to int32.
    V<Float64> number_value = ConvertHeapObjectToFloat64OrDeopt(
        input, frame_state, input_requirement, feedback);
    GOTO(done, __ JSTruncateFloat64ToWord32(number_value));

    BIND(done, result);
    return result;
  }

  V<Word32> JSAnyIsNotPrimitiveHeapObject(V<Object> value,
                                          V<Map> value_map = OpIndex{}) {
    if (!value_map.valid()) {
      value_map = __ LoadMapField(value);
    }
#if V8_STATIC_ROOTS_BOOL
    // Assumes only primitive objects and JS_RECEIVER's are passed here. All
    // primitive object's maps are in RO space and are allocated before all
    // JS_RECEIVER maps. Thus primitive object maps have smaller (compressed)
    // addresses.
    return __ Uint32LessThan(
        InstanceTypeChecker::kNonJsReceiverMapLimit,
        __ TruncateWordPtrToWord32(__ BitcastTaggedToWord(value_map)));
#else
    static_assert(LAST_TYPE == LAST_JS_RECEIVER_TYPE);
    V<Word32> value_instance_type = __ LoadInstanceTypeField(value_map);
    return __ Uint32LessThanOrEqual(FIRST_JS_RECEIVER_TYPE,
                                    value_instance_type);
#endif
  }

  OpIndex REDUCE(ConvertJSPrimitiveToObject)(V<Object> value,
                                             V<Object> global_proxy,
                                             ConvertReceiverMode mode) {
    switch (mode) {
      case ConvertReceiverMode::kNullOrUndefined:
        return global_proxy;
      case ConvertReceiverMode::kNotNullOrUndefined:
      case ConvertReceiverMode::kAny: {
        Label<Object> done(this);

        // Check if {value} is already a JSReceiver (or null/undefined).
        Label<> convert_to_object(this);
        GOTO_IF(UNLIKELY(__ ObjectIsSmi(value)), convert_to_object);
        GOTO_IF_NOT(LIKELY(__ JSAnyIsNotPrimitiveHeapObject(value)),
                    convert_to_object);
        GOTO(done, value);

        // Wrap the primitive {value} into a JSPrimitiveWrapper.
        if (BIND(convert_to_object)) {
          if (mode != ConvertReceiverMode::kNotNullOrUndefined) {
            // Replace the {value} with the {global_proxy}.
            GOTO_IF(UNLIKELY(__ TaggedEqual(
                        value, __ HeapConstant(factory_->undefined_value()))),
                    done, global_proxy);
            GOTO_IF(UNLIKELY(__ TaggedEqual(
                        value, __ HeapConstant(factory_->null_value()))),
                    done, global_proxy);
          }
          V<NativeContext> native_context =
              __ template LoadField<NativeContext>(
                  global_proxy, AccessBuilder::ForJSGlobalProxyNativeContext());
          GOTO(done, __ CallBuiltin_ToObject(isolate_, native_context, value));
        }

        BIND(done, result);
        return result;
      }
    }
    UNREACHABLE();
  }

  OpIndex REDUCE(NewConsString)(OpIndex length, OpIndex first, OpIndex second) {
    // Determine the instance types of {first} and {second}.
    V<Map> first_map = __ LoadMapField(first);
    V<Word32> first_type = __ LoadInstanceTypeField(first_map);
    V<Map> second_map = __ LoadMapField(second);
    V<Word32> second_type = __ LoadInstanceTypeField(second_map);

    Label<Tagged> allocate_string(this);
    // Determine the proper map for the resulting ConsString.
    // If both {first} and {second} are one-byte strings, we
    // create a new ConsOneByteString, otherwise we create a
    // new ConsString instead.
    static_assert(kOneByteStringTag != 0);
    static_assert(kTwoByteStringTag == 0);
    V<Word32> instance_type = __ Word32BitwiseAnd(first_type, second_type);
    V<Word32> encoding =
        __ Word32BitwiseAnd(instance_type, kStringEncodingMask);
    IF(__ Word32Equal(encoding, kTwoByteStringTag)) {
      GOTO(allocate_string,
           __ HeapConstant(factory_->cons_two_byte_string_map()));
    }
    ELSE {
      GOTO(allocate_string,
           __ HeapConstant(factory_->cons_one_byte_string_map()));
    }

    // Allocate the resulting ConsString.
    BIND(allocate_string, map);
    auto string = __ template Allocate<String>(
        __ IntPtrConstant(sizeof(ConsString)), AllocationType::kYoung);
    __ InitializeField(string, AccessBuilder::ForMap(), map);
    __ InitializeField(string, AccessBuilder::ForNameRawHashField(),
                       __ Word32Constant(Name::kEmptyHashField));
    __ InitializeField(string, AccessBuilder::ForStringLength(), length);
    __ InitializeField(string, AccessBuilder::ForConsStringFirst(), first);
    __ InitializeField(string, AccessBuilder::ForConsStringSecond(), second);
    return __ FinishInitialization(std::move(string));
  }

  OpIndex REDUCE(NewArray)(V<WordPtr> length, NewArrayOp::Kind kind,
                           AllocationType allocation_type) {
    Label<Tagged> done(this);

    GOTO_IF(__ WordPtrEqual(length, 0), done,
            __ HeapConstant(factory_->empty_fixed_array()));

    // Compute the effective size of the backing store.
    intptr_t size_log2;
    Handle<Map> array_map;
    // TODO(nicohartmann@): Replace ElementAccess by a Turboshaft replacement.
    ElementAccess access;
    V<Any> the_hole_value;
    switch (kind) {
      case NewArrayOp::Kind::kDouble: {
        size_log2 = kDoubleSizeLog2;
        array_map = factory_->fixed_double_array_map();
        access = {kTaggedBase, FixedDoubleArray::kHeaderSize,
                  compiler::Type::NumberOrHole(), MachineType::Float64(),
                  kNoWriteBarrier};
        the_hole_value = __ template LoadField<Float64>(
            __ HeapConstant(factory_->the_hole_value()),
            AccessBuilder::ForHeapNumberOrOddballValue());
        break;
      }
      case NewArrayOp::Kind::kObject: {
        size_log2 = kTaggedSizeLog2;
        array_map = factory_->fixed_array_map();
        access = {kTaggedBase, FixedArray::kHeaderSize, compiler::Type::Any(),
                  MachineType::AnyTagged(), kNoWriteBarrier};
        the_hole_value = __ HeapConstant(factory_->the_hole_value());
        break;
      }
    }
    V<WordPtr> size =
        __ WordPtrAdd(__ WordPtrShiftLeft(length, static_cast<int>(size_log2)),
                      access.header_size);

    // Allocate the result and initialize the header.
    auto uninitialized_array =
        __ template Allocate<FixedArray>(size, allocation_type);
    __ InitializeField(uninitialized_array, AccessBuilder::ForMap(),
                       __ HeapConstant(array_map));
    __ InitializeField(uninitialized_array,
                       AccessBuilder::ForFixedArrayLength(),
                       __ TagSmi(__ TruncateWordPtrToWord32(length)));
    // TODO(nicohartmann@): Should finish initialization only after all elements
    // have been initialized.
    auto array = __ FinishInitialization(std::move(uninitialized_array));

    // Initialize the backing store with holes.
    LoopLabel<WordPtr> loop(this);
    GOTO(loop, intptr_t{0});

    LOOP(loop, index) {
      GOTO_IF_NOT(LIKELY(__ UintPtrLessThan(index, length)), done, array);

      __ StoreNonArrayBufferElement(array, access, index, the_hole_value);

      // Advance the {index}.
      GOTO(loop, __ WordPtrAdd(index, 1));
    }

    BIND(done, result);
    return result;
  }

  OpIndex REDUCE(DoubleArrayMinMax)(V<Tagged> array,
                                    DoubleArrayMinMaxOp::Kind kind) {
    DCHECK(kind == DoubleArrayMinMaxOp::Kind::kMin ||
           kind == DoubleArrayMinMaxOp::Kind::kMax);
    const bool is_max = kind == DoubleArrayMinMaxOp::Kind::kMax;

    // Iterate the elements and find the result.
    V<Float64> empty_value =
        __ Float64Constant(is_max ? -V8_INFINITY : V8_INFINITY);
    V<WordPtr> array_length =
        __ ChangeInt32ToIntPtr(__ UntagSmi(__ template LoadField<Tagged>(
            array, AccessBuilder::ForJSArrayLength(
                       ElementsKind::PACKED_DOUBLE_ELEMENTS))));
    V<Tagged> elements = __ template LoadField<Tagged>(
        array, AccessBuilder::ForJSObjectElements());

    Label<> done(this);
    LoopLabel<WordPtr> loop(this);
    ScopedVar<Float64> result(Asm(), empty_value);

    GOTO(loop, intptr_t{0});

    LOOP(loop, index) {
      GOTO_IF_NOT(LIKELY(__ UintPtrLessThan(index, array_length)), done);

      V<Float64> element = __ template LoadNonArrayBufferElement<Float64>(
          elements, AccessBuilder::ForFixedDoubleArrayElement(), index);

      result = is_max ? __ Float64Max(*result, element)
                      : __ Float64Min(*result, element);
      GOTO(loop, __ WordPtrAdd(index, 1));
    }

    BIND(done);
    return __ ConvertFloat64ToNumber(*result,
                                     CheckForMinusZeroMode::kCheckForMinusZero);
  }

  OpIndex REDUCE(LoadFieldByIndex)(V<Tagged> object, V<Word32> field_index) {
    // Index encoding (see `src/objects/field-index-inl.h`):
    // For efficiency, the LoadByFieldIndex instruction takes an index that is
    // optimized for quick access. If the property is inline, the index is
    // positive. If it's out-of-line, the encoded index is -raw_index - 1 to
    // disambiguate the zero out-of-line index from the zero inobject case.
    // The index itself is shifted up by one bit, the lower-most bit
    // signifying if the field is a mutable double box (1) or not (0).
    V<WordPtr> index = __ ChangeInt32ToIntPtr(field_index);

    Label<> double_field(this);
    Label<Tagged> done(this);

    // Check if field is a mutable double field.
    GOTO_IF(
        UNLIKELY(__ TruncateWordPtrToWord32(__ WordPtrBitwiseAnd(index, 0x1))),
        double_field);

    {
      // The field is a proper Tagged field on {object}. The {index} is
      // shifted to the left by one in the code below.

      // Check if field is in-object or out-of-object.
      IF(__ IntPtrLessThan(index, 0)) {
        // The field is located in the properties backing store of {object}.
        // The {index} is equal to the negated out of property index plus 1.
        V<Tagged> properties = __ template LoadField<Tagged>(
            object, AccessBuilder::ForJSObjectPropertiesOrHashKnownPointer());

        V<WordPtr> out_of_object_index = __ WordPtrSub(0, index);
        V<Tagged> result =
            __ Load(properties, out_of_object_index,
                    LoadOp::Kind::Aligned(BaseTaggedness::kTaggedBase),
                    MemoryRepresentation::AnyTagged(),
                    FixedArray::kHeaderSize - kTaggedSize, kTaggedSizeLog2 - 1);
        GOTO(done, result);
      }
      ELSE {
        // This field is located in the {object} itself.
        V<Tagged> result = __ Load(
            object, index, LoadOp::Kind::Aligned(BaseTaggedness::kTaggedBase),
            MemoryRepresentation::AnyTagged(), JSObject::kHeaderSize,
            kTaggedSizeLog2 - 1);
        GOTO(done, result);
      }
      END_IF
    }

    if (BIND(double_field)) {
      // If field is a Double field, either unboxed in the object on 64 bit
      // architectures, or a mutable HeapNumber.
      V<WordPtr> double_index = __ WordPtrShiftRightArithmetic(index, 1);
      Label<Tagged> loaded_field(this);

      // Check if field is in-object or out-of-object.
      IF(__ IntPtrLessThan(double_index, 0)) {
        V<Tagged> properties = __ template LoadField<Tagged>(
            object, AccessBuilder::ForJSObjectPropertiesOrHashKnownPointer());

        V<WordPtr> out_of_object_index = __ WordPtrSub(0, double_index);
        V<Tagged> result =
            __ Load(properties, out_of_object_index,
                    LoadOp::Kind::Aligned(BaseTaggedness::kTaggedBase),
                    MemoryRepresentation::AnyTagged(),
                    FixedArray::kHeaderSize - kTaggedSize, kTaggedSizeLog2);
        GOTO(loaded_field, result);
      }
      ELSE {
        // The field is located in the {object} itself.
        V<Tagged> result =
            __ Load(object, double_index,
                    LoadOp::Kind::Aligned(BaseTaggedness::kTaggedBase),
                    MemoryRepresentation::AnyTagged(), JSObject::kHeaderSize,
                    kTaggedSizeLog2);
        GOTO(loaded_field, result);
      }
      END_IF

      if (BIND(loaded_field, field)) {
        // We may have transitioned in-place away from double, so check that
        // this is a HeapNumber -- otherwise the load is fine and we don't need
        // to copy anything anyway.
        GOTO_IF(__ ObjectIsSmi(field), done, field);
        V<Map> map = __ LoadMapField(field);
        GOTO_IF_NOT(
            __ TaggedEqual(map, __ HeapConstant(factory_->heap_number_map())),
            done, field);

        V<Float64> value = __ template LoadField<Float64>(
            field, AccessBuilder::ForHeapNumberValue());
        GOTO(done, AllocateHeapNumberWithValue(value));
      }
    }

    BIND(done, result);
    return result;
  }

  OpIndex REDUCE(BigIntBinop)(V<Tagged> left, V<Tagged> right,
                              OpIndex frame_state, BigIntBinopOp::Kind kind) {
    const Builtin builtin = GetBuiltinForBigIntBinop(kind);
    switch (kind) {
      case BigIntBinopOp::Kind::kAdd:
      case BigIntBinopOp::Kind::kSub:
      case BigIntBinopOp::Kind::kBitwiseAnd:
      case BigIntBinopOp::Kind::kBitwiseXor:
      case BigIntBinopOp::Kind::kShiftLeft:
      case BigIntBinopOp::Kind::kShiftRightArithmetic: {
        V<Tagged> result = CallBuiltinForBigIntOp(builtin, {left, right});

        // Check for exception sentinel: Smi 0 is returned to signal
        // BigIntTooBig.
        __ DeoptimizeIf(__ ObjectIsSmi(result), frame_state,
                        DeoptimizeReason::kBigIntTooBig, FeedbackSource{});
        return result;
      }
      case BigIntBinopOp::Kind::kMul:
      case BigIntBinopOp::Kind::kDiv:
      case BigIntBinopOp::Kind::kMod: {
        V<Tagged> result = CallBuiltinForBigIntOp(builtin, {left, right});

        // Check for exception sentinel: Smi 1 is returned to signal
        // TerminationRequested.
        IF (UNLIKELY(__ TaggedEqual(result, __ TagSmi(1)))) {
          __ CallRuntime_TerminateExecution(isolate_, frame_state,
                                            __ NoContextConstant());
        }
        END_IF

        // Check for exception sentinel: Smi 0 is returned to signal
        // BigIntTooBig or DivisionByZero.
        __ DeoptimizeIf(__ ObjectIsSmi(result), frame_state,
                        kind == BigIntBinopOp::Kind::kMul
                            ? DeoptimizeReason::kBigIntTooBig
                            : DeoptimizeReason::kDivisionByZero,
                        FeedbackSource{});
        return result;
      }
      case BigIntBinopOp::Kind::kBitwiseOr: {
        return CallBuiltinForBigIntOp(builtin, {left, right});
      }
      default:
        UNIMPLEMENTED();
    }
    UNREACHABLE();
  }

  V<Boolean> REDUCE(BigIntComparison)(V<Tagged> left, V<Tagged> right,
                                      BigIntComparisonOp::Kind kind) {
    switch (kind) {
      case BigIntComparisonOp::Kind::kEqual:
        return CallBuiltinForBigIntOp(Builtin::kBigIntEqual, {left, right});
      case BigIntComparisonOp::Kind::kLessThan:
        return CallBuiltinForBigIntOp(Builtin::kBigIntLessThan, {left, right});
      case BigIntComparisonOp::Kind::kLessThanOrEqual:
        return CallBuiltinForBigIntOp(Builtin::kBigIntLessThanOrEqual,
                                      {left, right});
    }
  }

  V<Tagged> REDUCE(BigIntUnary)(V<Tagged> input, BigIntUnaryOp::Kind kind) {
    DCHECK_EQ(kind, BigIntUnaryOp::Kind::kNegate);
    return CallBuiltinForBigIntOp(Builtin::kBigIntUnaryMinus, {input});
  }

  V<Word32> REDUCE(StringAt)(V<String> string, V<WordPtr> pos,
                             StringAtOp::Kind kind) {
    if (kind == StringAtOp::Kind::kCharCode) {
      Label<Word32> done(this);
      // TODO(dmercadier): the runtime label should be deferred, and because
      // Labels/Blocks don't have deferred annotation, we achieve this by
      // marking all branches to this Label as UNLIKELY, but 1) it's easy to
      // forget one, and 2) it makes the code less clear: `if(x) {} else
      // if(likely(y)) {} else {}` looks like `y` is more likely than `x`, but
      // it just means that `y` is more likely than `!y`.
      Label<> runtime(this);
      // We need a loop here to properly deal with indirect strings
      // (SlicedString, ConsString and ThinString).
      LoopLabel<> loop(this);
      ScopedVar<String> receiver(Asm(), string);
      ScopedVar<WordPtr> position(Asm(), pos);
      GOTO(loop);

      LOOP(loop) {
        V<Map> map = __ LoadMapField(*receiver);
        V<Word32> instance_type = __ LoadInstanceTypeField(map);
        V<Word32> representation =
            __ Word32BitwiseAnd(instance_type, kStringRepresentationMask);

        IF (__ Int32LessThanOrEqual(representation, kConsStringTag)) {
          {
            // if_lessthanoreq_cons
            IF (__ Word32Equal(representation, kConsStringTag)) {
              // if_consstring
              V<String> second = __ template LoadField<String>(
                  *receiver, AccessBuilder::ForConsStringSecond());
              GOTO_IF_NOT(
                  LIKELY(__ TaggedEqual(
                      second, __ HeapConstant(factory_->empty_string()))),
                  runtime);
              receiver = __ template LoadField<String>(
                  *receiver, AccessBuilder::ForConsStringFirst());
              GOTO(loop);
            }
            ELSE {
              // if_seqstring
              V<Word32> onebyte = __ Word32Equal(
                  __ Word32BitwiseAnd(instance_type, kStringEncodingMask),
                  kOneByteStringTag);
              GOTO(done, LoadFromSeqString(*receiver, *position, onebyte));
            }
            END_IF
          }
        }
        ELSE {
          // if_greaterthan_cons
          {
            IF (__ Word32Equal(representation, kThinStringTag)) {
              // if_thinstring
              receiver = __ template LoadField<String>(
                  *receiver, AccessBuilder::ForThinStringActual());
              GOTO(loop);
            }
            ELSE_IF (__ Word32Equal(representation, kExternalStringTag)) {
              // if_externalstring
              // We need to bailout to the runtime for uncached external
              // strings.
              GOTO_IF(UNLIKELY(__ Word32Equal(
                          __ Word32BitwiseAnd(instance_type,
                                              kUncachedExternalStringMask),
                          kUncachedExternalStringTag)),
                      runtime);

              OpIndex data = __ LoadField(
                  *receiver, AccessBuilder::ForExternalStringResourceData());
              IF (__ Word32Equal(
                      __ Word32BitwiseAnd(instance_type, kStringEncodingMask),
                      kTwoByteStringTag)) {
                // if_twobyte
                constexpr uint8_t twobyte_size_log2 = 1;
                V<Word32> value = __ Load(
                    data, *position,
                    LoadOp::Kind::Aligned(BaseTaggedness::kUntaggedBase),
                    MemoryRepresentation::Uint16(), 0, twobyte_size_log2);
                GOTO(done, value);
              }
              ELSE {
                // if_onebyte
                constexpr uint8_t onebyte_size_log2 = 0;
                V<Word32> value = __ Load(
                    data, *position,
                    LoadOp::Kind::Aligned(BaseTaggedness::kUntaggedBase),
                    MemoryRepresentation::Uint8(), 0, onebyte_size_log2);
                GOTO(done, value);
              }
              END_IF
            }
            ELSE_IF (LIKELY(__ Word32Equal(representation, kSlicedStringTag))) {
              // if_slicedstring
              V<Tagged> offset = __ template LoadField<Tagged>(
                  *receiver, AccessBuilder::ForSlicedStringOffset());
              receiver = __ template LoadField<String>(
                  *receiver, AccessBuilder::ForSlicedStringParent());
              position = __ WordPtrAdd(
                  *position, __ ChangeInt32ToIntPtr(__ UntagSmi(offset)));
              GOTO(loop);
            }
            ELSE {
              GOTO(runtime);
            }
            END_IF
          }
        }
        END_IF

        if (BIND(runtime)) {
          V<Word32> value = __ UntagSmi(__ CallRuntime_StringCharCodeAt(
              isolate_, __ NoContextConstant(), *receiver,
              __ TagSmi(__ TruncateWordPtrToWord32(*position))));
          GOTO(done, value);
        }
      }

      BIND(done, result);
      return result;
    } else {
      DCHECK_EQ(kind, StringAtOp::Kind::kCodePoint);
      Label<Word32> done(this);

      V<Word32> first_code_unit = __ StringCharCodeAt(string, pos);
      GOTO_IF_NOT(UNLIKELY(__ Word32Equal(
                      __ Word32BitwiseAnd(first_code_unit, 0xFC00), 0xD800)),
                  done, first_code_unit);
      V<WordPtr> length =
          __ ChangeUint32ToUintPtr(__ template LoadField<Word32>(
              string, AccessBuilder::ForStringLength()));
      V<WordPtr> next_index = __ WordPtrAdd(pos, 1);
      GOTO_IF_NOT(__ IntPtrLessThan(next_index, length), done, first_code_unit);

      V<Word32> second_code_unit = __ StringCharCodeAt(string, next_index);
      GOTO_IF_NOT(
          __ Word32Equal(__ Word32BitwiseAnd(second_code_unit, 0xFC00), 0xDC00),
          done, first_code_unit);

      const int32_t surrogate_offset = 0x10000 - (0xD800 << 10) - 0xDC00;
      V<Word32> value =
          __ Word32Add(__ Word32ShiftLeft(first_code_unit, 10),
                       __ Word32Add(second_code_unit, surrogate_offset));
      GOTO(done, value);

      BIND(done, result);
      return result;
    }
