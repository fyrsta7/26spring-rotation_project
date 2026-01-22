template <class Next>
class MachineOptimizationReducer : public Next {
 public:
  TURBOSHAFT_REDUCER_BOILERPLATE()

  // TODO(mslekova): Implement ReduceSelect and ReducePhi,
  // by reducing `(f > 0) ? f : -f` to `fabs(f)`.

  OpIndex REDUCE(Change)(OpIndex input, ChangeOp::Kind kind,
                         ChangeOp::Assumption assumption,
                         RegisterRepresentation from,
                         RegisterRepresentation to) {
    if (ShouldSkipOptimizationStep()) {
      return Next::ReduceChange(input, kind, assumption, from, to);
    }
    using Kind = ChangeOp::Kind;
    if (from == WordRepresentation::Word32()) {
      input = TryRemoveWord32ToWord64Conversion(input);
    }
    if (uint64_t value;
        from.IsWord() && matcher.MatchIntegralWordConstant(
                             input, WordRepresentation(from), &value)) {
      using Rep = RegisterRepresentation;
      switch (multi(kind, from, to)) {
        case multi(Kind::kSignExtend, Rep::Word32(), Rep::Word64()):
          return __ Word64Constant(int64_t{static_cast<int32_t>(value)});
        case multi(Kind::kZeroExtend, Rep::Word32(), Rep::Word64()):
        case multi(Kind::kBitcast, Rep::Word32(), Rep::Word64()):
          return __ Word64Constant(uint64_t{static_cast<uint32_t>(value)});
        case multi(Kind::kBitcast, Rep::Word32(), Rep::Float32()):
          return __ Float32Constant(
              base::bit_cast<float>(static_cast<uint32_t>(value)));
        case multi(Kind::kBitcast, Rep::Word64(), Rep::Float64()):
          return __ Float64Constant(base::bit_cast<double>(value));
        case multi(Kind::kSignedToFloat, Rep::Word32(), Rep::Float64()):
          return __ Float64Constant(
              static_cast<double>(static_cast<int32_t>(value)));
        case multi(Kind::kSignedToFloat, Rep::Word64(), Rep::Float64()):
          return __ Float64Constant(
              static_cast<double>(static_cast<int64_t>(value)));
        case multi(Kind::kUnsignedToFloat, Rep::Word32(), Rep::Float64()):
          return __ Float64Constant(
              static_cast<double>(static_cast<uint32_t>(value)));
        case multi(Kind::kTruncate, Rep::Word64(), Rep::Word32()):
          return __ Word32Constant(static_cast<uint32_t>(value));
        default:
          break;
      }
    }
    if (float value; from == RegisterRepresentation::Float32() &&
                     matcher.MatchFloat32Constant(input, &value)) {
      if (kind == Kind::kFloatConversion &&
          to == RegisterRepresentation::Float64()) {
        return __ Float64Constant(value);
      }
      if (kind == Kind::kBitcast && to == WordRepresentation::Word32()) {
        return __ Word32Constant(base::bit_cast<uint32_t>(value));
      }
    }
    if (double value; from == RegisterRepresentation::Float64() &&
                      matcher.MatchFloat64Constant(input, &value)) {
      if (kind == Kind::kFloatConversion &&
          to == RegisterRepresentation::Float32()) {
        return __ Float32Constant(DoubleToFloat32_NoInline(value));
      }
      if (kind == Kind::kBitcast && to == WordRepresentation::Word64()) {
        return __ Word64Constant(base::bit_cast<uint64_t>(value));
      }
      if (kind == Kind::kSignedFloatTruncateOverflowToMin) {
        double truncated = std::trunc(value);
        if (to == WordRepresentation::Word64()) {
          int64_t result = std::numeric_limits<int64_t>::min();
          if (truncated >= std::numeric_limits<int64_t>::min() &&
              truncated <= kMaxDoubleRepresentableInt64) {
            result = static_cast<int64_t>(truncated);
          }
          return __ Word64Constant(result);
        }
        if (to == WordRepresentation::Word32()) {
          int32_t result = std::numeric_limits<int32_t>::min();
          if (truncated >= std::numeric_limits<int32_t>::min() &&
              truncated <= std::numeric_limits<int32_t>::max()) {
            result = static_cast<int32_t>(truncated);
          }
          return __ Word32Constant(result);
        }
      }
      if (kind == Kind::kJSFloatTruncate &&
          to == WordRepresentation::Word32()) {
        return __ Word32Constant(DoubleToInt32_NoInline(value));
      }
      if (kind == Kind::kExtractHighHalf) {
        DCHECK_EQ(to, RegisterRepresentation::Word32());
        return __ Word32Constant(
            static_cast<uint32_t>(base::bit_cast<uint64_t>(value) >> 32));
      }
      if (kind == Kind::kExtractLowHalf) {
        DCHECK_EQ(to, RegisterRepresentation::Word32());
        return __ Word32Constant(
            static_cast<uint32_t>(base::bit_cast<uint64_t>(value)));
      }
    }
    if (float value; from == RegisterRepresentation::Float32() &&
                     matcher.MatchFloat32Constant(input, &value)) {
      if (kind == Kind::kFloatConversion &&
          to == RegisterRepresentation::Float64()) {
        return __ Float64Constant(value);
      }
    }

    const Operation& input_op = matcher.Get(input);
    if (const ChangeOp* change_op = input_op.TryCast<ChangeOp>()) {
      if (change_op->from == to && change_op->to == from &&
          change_op->IsReversibleBy(kind, signalling_nan_possible)) {
        return change_op->input();
      }
    }
    return Next::ReduceChange(input, kind, assumption, from, to);
  }

  OpIndex REDUCE(BitcastWord32PairToFloat64)(OpIndex hi_word32,
                                             OpIndex lo_word32) {
    if (ShouldSkipOptimizationStep()) {
      return Next::ReduceBitcastWord32PairToFloat64(hi_word32, lo_word32);
    }
    uint32_t lo, hi;
    if (matcher.MatchIntegralWord32Constant(hi_word32, &hi) &&
        matcher.MatchIntegralWord32Constant(lo_word32, &lo)) {
      return __ Float64Constant(
          base::bit_cast<double>(uint64_t{hi} << 32 | uint64_t{lo}));
    }
    return Next::ReduceBitcastWord32PairToFloat64(hi_word32, lo_word32);
  }

  OpIndex REDUCE(TaggedBitcast)(OpIndex input, RegisterRepresentation from,
                                RegisterRepresentation to) {
    if (ShouldSkipOptimizationStep()) {
      return Next::ReduceTaggedBitcast(input, from, to);
    }
    // A Tagged -> Untagged -> Tagged sequence can be short-cut.
    // An Untagged -> Tagged -> Untagged sequence however cannot be removed,
    // because the GC might have modified the pointer.
    if (auto* input_bitcast = matcher.TryCast<TaggedBitcastOp>(input)) {
      if (all_of(input_bitcast->to, from) ==
              RegisterRepresentation::PointerSized() &&
          all_of(input_bitcast->from, to) == RegisterRepresentation::Tagged()) {
        return input_bitcast->input();
      }
    }
    // Try to constant-fold TaggedBitcast from Word Constant to Word.
    if (to.IsWord()) {
      if (const ConstantOp* cst = matcher.TryCast<ConstantOp>(input)) {
        if (cst->kind == ConstantOp::Kind::kWord32 ||
            cst->kind == ConstantOp::Kind::kWord64) {
          if (to == RegisterRepresentation::Word64()) {
            return __ Word64Constant(cst->integral());
          } else {
            DCHECK_EQ(to, RegisterRepresentation::Word32());
            return __ Word32Constant(static_cast<uint32_t>(cst->integral()));
          }
        }
      }
    }
    return Next::ReduceTaggedBitcast(input, from, to);
  }

  OpIndex REDUCE(FloatUnary)(OpIndex input, FloatUnaryOp::Kind kind,
                             FloatRepresentation rep) {
    if (ShouldSkipOptimizationStep()) {
      return Next::ReduceFloatUnary(input, kind, rep);
    }
    if (float k; rep == FloatRepresentation::Float32() &&
                 matcher.MatchFloat32Constant(input, &k)) {
      if (std::isnan(k) && !signalling_nan_possible) {
        return __ Float32Constant(std::numeric_limits<float>::quiet_NaN());
      }
      switch (kind) {
        case FloatUnaryOp::Kind::kAbs:
          return __ Float32Constant(std::abs(k));
        case FloatUnaryOp::Kind::kNegate:
          return __ Float32Constant(-k);
        case FloatUnaryOp::Kind::kSilenceNaN:
          DCHECK(!std::isnan(k));
          return __ Float32Constant(k);
        case FloatUnaryOp::Kind::kRoundDown:
          return __ Float32Constant(std::floor(k));
        case FloatUnaryOp::Kind::kRoundUp:
          return __ Float32Constant(std::ceil(k));
        case FloatUnaryOp::Kind::kRoundToZero:
          return __ Float32Constant(std::trunc(k));
        case FloatUnaryOp::Kind::kRoundTiesEven:
          DCHECK_EQ(std::nearbyint(1.5), 2);
          DCHECK_EQ(std::nearbyint(2.5), 2);
          return __ Float32Constant(std::nearbyint(k));
        case FloatUnaryOp::Kind::kLog:
          return __ Float32Constant(base::ieee754::log(k));
        case FloatUnaryOp::Kind::kSqrt:
          return __ Float32Constant(std::sqrt(k));
        case FloatUnaryOp::Kind::kExp:
          return __ Float32Constant(base::ieee754::exp(k));
        case FloatUnaryOp::Kind::kExpm1:
          return __ Float32Constant(base::ieee754::expm1(k));
        case FloatUnaryOp::Kind::kSin:
          return __ Float32Constant(SIN_IMPL(k));
        case FloatUnaryOp::Kind::kCos:
          return __ Float32Constant(COS_IMPL(k));
        case FloatUnaryOp::Kind::kSinh:
          return __ Float32Constant(base::ieee754::sinh(k));
        case FloatUnaryOp::Kind::kCosh:
          return __ Float32Constant(base::ieee754::cosh(k));
        case FloatUnaryOp::Kind::kAcos:
          return __ Float32Constant(base::ieee754::acos(k));
        case FloatUnaryOp::Kind::kAsin:
          return __ Float32Constant(base::ieee754::asin(k));
        case FloatUnaryOp::Kind::kAsinh:
          return __ Float32Constant(base::ieee754::asinh(k));
        case FloatUnaryOp::Kind::kAcosh:
          return __ Float32Constant(base::ieee754::acosh(k));
        case FloatUnaryOp::Kind::kTan:
          return __ Float32Constant(base::ieee754::tan(k));
        case FloatUnaryOp::Kind::kTanh:
          return __ Float32Constant(base::ieee754::tanh(k));
        case FloatUnaryOp::Kind::kLog2:
          return __ Float32Constant(base::ieee754::log2(k));
        case FloatUnaryOp::Kind::kLog10:
          return __ Float32Constant(base::ieee754::log10(k));
        case FloatUnaryOp::Kind::kLog1p:
          return __ Float32Constant(base::ieee754::log1p(k));
        case FloatUnaryOp::Kind::kCbrt:
          return __ Float32Constant(base::ieee754::cbrt(k));
        case FloatUnaryOp::Kind::kAtan:
          return __ Float32Constant(base::ieee754::atan(k));
        case FloatUnaryOp::Kind::kAtanh:
          return __ Float32Constant(base::ieee754::atanh(k));
      }
    } else if (double k; rep == FloatRepresentation::Float64() &&
                         matcher.MatchFloat64Constant(input, &k)) {
      if (std::isnan(k)) {
        return __ Float64Constant(std::numeric_limits<double>::quiet_NaN());
      }
      switch (kind) {
        case FloatUnaryOp::Kind::kAbs:
          return __ Float64Constant(std::abs(k));
        case FloatUnaryOp::Kind::kNegate:
          return __ Float64Constant(-k);
        case FloatUnaryOp::Kind::kSilenceNaN:
          DCHECK(!std::isnan(k));
          return __ Float64Constant(k);
        case FloatUnaryOp::Kind::kRoundDown:
          return __ Float64Constant(std::floor(k));
        case FloatUnaryOp::Kind::kRoundUp:
          return __ Float64Constant(std::ceil(k));
        case FloatUnaryOp::Kind::kRoundToZero:
          return __ Float64Constant(std::trunc(k));
        case FloatUnaryOp::Kind::kRoundTiesEven:
          DCHECK_EQ(std::nearbyint(1.5), 2);
          DCHECK_EQ(std::nearbyint(2.5), 2);
          return __ Float64Constant(std::nearbyint(k));
        case FloatUnaryOp::Kind::kLog:
          return __ Float64Constant(base::ieee754::log(k));
        case FloatUnaryOp::Kind::kSqrt:
          return __ Float64Constant(std::sqrt(k));
        case FloatUnaryOp::Kind::kExp:
          return __ Float64Constant(base::ieee754::exp(k));
        case FloatUnaryOp::Kind::kExpm1:
          return __ Float64Constant(base::ieee754::expm1(k));
        case FloatUnaryOp::Kind::kSin:
          return __ Float64Constant(SIN_IMPL(k));
        case FloatUnaryOp::Kind::kCos:
          return __ Float64Constant(COS_IMPL(k));
        case FloatUnaryOp::Kind::kSinh:
          return __ Float64Constant(base::ieee754::sinh(k));
        case FloatUnaryOp::Kind::kCosh:
          return __ Float64Constant(base::ieee754::cosh(k));
        case FloatUnaryOp::Kind::kAcos:
          return __ Float64Constant(base::ieee754::acos(k));
        case FloatUnaryOp::Kind::kAsin:
          return __ Float64Constant(base::ieee754::asin(k));
        case FloatUnaryOp::Kind::kAsinh:
          return __ Float64Constant(base::ieee754::asinh(k));
        case FloatUnaryOp::Kind::kAcosh:
          return __ Float64Constant(base::ieee754::acosh(k));
        case FloatUnaryOp::Kind::kTan:
          return __ Float64Constant(base::ieee754::tan(k));
        case FloatUnaryOp::Kind::kTanh:
          return __ Float64Constant(base::ieee754::tanh(k));
        case FloatUnaryOp::Kind::kLog2:
          return __ Float64Constant(base::ieee754::log2(k));
        case FloatUnaryOp::Kind::kLog10:
          return __ Float64Constant(base::ieee754::log10(k));
        case FloatUnaryOp::Kind::kLog1p:
          return __ Float64Constant(base::ieee754::log1p(k));
        case FloatUnaryOp::Kind::kCbrt:
          return __ Float64Constant(base::ieee754::cbrt(k));
        case FloatUnaryOp::Kind::kAtan:
          return __ Float64Constant(base::ieee754::atan(k));
        case FloatUnaryOp::Kind::kAtanh:
          return __ Float64Constant(base::ieee754::atanh(k));
      }
    }
    return Next::ReduceFloatUnary(input, kind, rep);
  }

  OpIndex REDUCE(WordUnary)(OpIndex input, WordUnaryOp::Kind kind,
                            WordRepresentation rep) {
    if (ShouldSkipOptimizationStep()) {
      return Next::ReduceWordUnary(input, kind, rep);
    }
    if (rep == WordRepresentation::Word32()) {
      input = TryRemoveWord32ToWord64Conversion(input);
    }
    if (uint32_t k; rep == WordRepresentation::Word32() &&
                    matcher.MatchIntegralWord32Constant(input, &k)) {
      switch (kind) {
        case WordUnaryOp::Kind::kReverseBytes:
          return __ Word32Constant(base::bits::ReverseBytes(k));
        case WordUnaryOp::Kind::kCountLeadingZeros:
          return __ Word32Constant(base::bits::CountLeadingZeros(k));
        case WordUnaryOp::Kind::kCountTrailingZeros:
          return __ Word32Constant(base::bits::CountTrailingZeros(k));
        case WordUnaryOp::Kind::kPopCount:
          return __ Word32Constant(base::bits::CountPopulation(k));
        case WordUnaryOp::Kind::kSignExtend8:
          return __ Word32Constant(int32_t{static_cast<int8_t>(k)});
        case WordUnaryOp::Kind::kSignExtend16:
          return __ Word32Constant(int32_t{static_cast<int16_t>(k)});
      }
    } else if (uint64_t k; rep == WordRepresentation::Word64() &&
                           matcher.MatchIntegralWord64Constant(input, &k)) {
      switch (kind) {
        case WordUnaryOp::Kind::kReverseBytes:
          return __ Word64Constant(base::bits::ReverseBytes(k));
        case WordUnaryOp::Kind::kCountLeadingZeros:
          return __ Word64Constant(uint64_t{base::bits::CountLeadingZeros(k)});
        case WordUnaryOp::Kind::kCountTrailingZeros:
          return __ Word64Constant(uint64_t{base::bits::CountTrailingZeros(k)});
        case WordUnaryOp::Kind::kPopCount:
          return __ Word64Constant(uint64_t{base::bits::CountPopulation(k)});
        case WordUnaryOp::Kind::kSignExtend8:
          return __ Word64Constant(int64_t{static_cast<int8_t>(k)});
        case WordUnaryOp::Kind::kSignExtend16:
          return __ Word64Constant(int64_t{static_cast<int16_t>(k)});
      }
    }
    return Next::ReduceWordUnary(input, kind, rep);
  }

  OpIndex REDUCE(FloatBinop)(OpIndex lhs, OpIndex rhs, FloatBinopOp::Kind kind,
                             FloatRepresentation rep) {
    if (ShouldSkipOptimizationStep()) {
      return Next::ReduceFloatBinop(lhs, rhs, kind, rep);
    }

    using Kind = FloatBinopOp::Kind;

    // Place constant on the right for commutative operators.
    if (FloatBinopOp::IsCommutative(kind) && matcher.Is<ConstantOp>(lhs) &&
        !matcher.Is<ConstantOp>(rhs)) {
      return ReduceFloatBinop(rhs, lhs, kind, rep);
    }

    // constant folding
    if (float k1, k2; rep == FloatRepresentation::Float32() &&
                      matcher.MatchFloat32Constant(lhs, &k1) &&
                      matcher.MatchFloat32Constant(rhs, &k2)) {
      switch (kind) {
        case Kind::kAdd:
          return __ Float32Constant(k1 + k2);
        case Kind::kMul:
          return __ Float32Constant(k1 * k2);
        case Kind::kSub:
          return __ Float32Constant(k1 - k2);
        case Kind::kMin:
          return __ Float32Constant(JSMin(k1, k2));
        case Kind::kMax:
          return __ Float32Constant(JSMax(k1, k2));
        case Kind::kDiv:
          return __ Float32Constant(k1 / k2);
        case Kind::kPower:
          return __ Float32Constant(base::ieee754::pow(k1, k2));
        case Kind::kAtan2:
          return __ Float32Constant(base::ieee754::atan2(k1, k2));
        case Kind::kMod:
          UNREACHABLE();
      }
    }
    if (double k1, k2; rep == FloatRepresentation::Float64() &&
                       matcher.MatchFloat64Constant(lhs, &k1) &&
                       matcher.MatchFloat64Constant(rhs, &k2)) {
      switch (kind) {
        case Kind::kAdd:
          return __ Float64Constant(k1 + k2);
        case Kind::kMul:
          return __ Float64Constant(k1 * k2);
        case Kind::kSub:
          return __ Float64Constant(k1 - k2);
        case Kind::kMin:
          return __ Float64Constant(JSMin(k1, k2));
        case Kind::kMax:
          return __ Float64Constant(JSMax(k1, k2));
        case Kind::kDiv:
          return __ Float64Constant(k1 / k2);
        case Kind::kMod:
          return __ Float64Constant(Modulo(k1, k2));
        case Kind::kPower:
          return __ Float64Constant(base::ieee754::pow(k1, k2));
        case Kind::kAtan2:
          return __ Float64Constant(base::ieee754::atan2(k1, k2));
      }
    }

    // lhs <op> NaN  =>  NaN
    if (matcher.MatchNaN(rhs) ||
        (matcher.MatchNaN(lhs) && kind != Kind::kPower)) {
      // Return a quiet NaN since Wasm operations could have signalling NaN as
      // input but not as output.
      return __ FloatConstant(std::numeric_limits<double>::quiet_NaN(), rep);
    }

    if (matcher.Is<ConstantOp>(rhs)) {
      if (kind == Kind::kMul) {
        // lhs * 1  =>  lhs
        if (!signalling_nan_possible && matcher.MatchFloat(rhs, 1.0)) {
          return lhs;
        }
        // lhs * 2  =>  lhs + lhs
        if (matcher.MatchFloat(rhs, 2.0)) {
          return __ FloatAdd(lhs, lhs, rep);
        }
        // lhs * -1  =>  -lhs
        if (!signalling_nan_possible && matcher.MatchFloat(rhs, -1.0)) {
          return __ FloatNegate(lhs, rep);
        }
      }

      if (kind == Kind::kDiv) {
        // lhs / 1  =>  lhs
        if (!signalling_nan_possible && matcher.MatchFloat(rhs, 1.0)) {
          return lhs;
        }
        // lhs / -1  =>  -lhs
        if (!signalling_nan_possible && matcher.MatchFloat(rhs, -1.0)) {
          return __ FloatNegate(lhs, rep);
        }
        // All reciprocals of non-denormal powers of two can be represented
        // exactly, so division by power of two can be reduced to
        // multiplication by reciprocal, with the same result.
        // x / k  =>  x * (1 / k)
        if (rep == FloatRepresentation::Float32()) {
          if (float k;
              matcher.MatchFloat32Constant(rhs, &k) && std::isnormal(k) &&
              k != 0 && std::isfinite(k) &&
              base::bits::IsPowerOfTwo(base::Double(k).Significand())) {
            return __ FloatMul(lhs, __ FloatConstant(1.0 / k, rep), rep);
          }
        } else {
          DCHECK_EQ(rep, FloatRepresentation::Float64());
          if (double k;
              matcher.MatchFloat64Constant(rhs, &k) && std::isnormal(k) &&
              k != 0 && std::isfinite(k) &&
              base::bits::IsPowerOfTwo(base::Double(k).Significand())) {
            return __ FloatMul(lhs, __ FloatConstant(1.0 / k, rep), rep);
          }
        }
      }

      if (kind == Kind::kMod) {
        // x % 0  =>  NaN
        if (matcher.MatchFloat(rhs, 0.0)) {
          return __ FloatConstant(std::numeric_limits<double>::quiet_NaN(),
                                  rep);
        }
      }

      if (kind == Kind::kSub) {
        // lhs - +0.0  =>  lhs
        if (!signalling_nan_possible && matcher.MatchFloat(rhs, +0.0)) {
          return lhs;
        }
      }

      if (kind == Kind::kPower) {
        if (matcher.MatchFloat(rhs, 0.0) || matcher.MatchFloat(rhs, -0.0)) {
          // lhs ** 0  ==>  1
          return __ FloatConstant(1.0, rep);
        }
        if (matcher.MatchFloat(rhs, 2.0)) {
          // lhs ** 2  ==>  lhs * lhs
          return __ FloatMul(lhs, lhs, rep);
        }
        if (matcher.MatchFloat(rhs, 0.5)) {
          // lhs ** 0.5  ==>  sqrt(lhs)
          // (unless if lhs is -infinity)
          Variable result = __ NewLoopInvariantVariable(rep);
          IF (UNLIKELY(__ FloatLessThanOrEqual(
                  lhs, __ FloatConstant(-V8_INFINITY, rep), rep))) {
            __ SetVariable(result, __ FloatConstant(V8_INFINITY, rep));
          }
          ELSE {
            __ SetVariable(result, __ FloatSqrt(lhs, rep));
          }
          END_IF
          return __ GetVariable(result);
        }
      }
    }

    if (!signalling_nan_possible && kind == Kind::kSub &&
        matcher.MatchFloat(lhs, -0.0)) {
      // -0.0 - round_down(-0.0 - y) => round_up(y)
      if (OpIndex a, b, c;
          FloatUnaryOp::IsSupported(FloatUnaryOp::Kind::kRoundUp, rep) &&
          matcher.MatchFloatRoundDown(rhs, &a, rep) &&
          matcher.MatchFloatSub(a, &b, &c, rep) &&
          matcher.MatchFloat(b, -0.0)) {
        return __ FloatRoundUp(c, rep);
      }
      // -0.0 - rhs  =>  -rhs
      return __ FloatNegate(rhs, rep);
    }

    return Next::ReduceFloatBinop(lhs, rhs, kind, rep);
  }

  OpIndex REDUCE(WordBinop)(OpIndex left, OpIndex right, WordBinopOp::Kind kind,
                            WordRepresentation rep) {
    if (ShouldSkipOptimizationStep()) {
      return Next::ReduceWordBinop(left, right, kind, rep);
    }

    using Kind = WordBinopOp::Kind;

    DCHECK_EQ(rep, any_of(WordRepresentation::Word32(),
                          WordRepresentation::Word64()));
    bool is_64 = rep == WordRepresentation::Word64();

    if (!is_64) {
      left = TryRemoveWord32ToWord64Conversion(left);
      right = TryRemoveWord32ToWord64Conversion(right);
    }

    // Place constant on the right for commutative operators.
    if (WordBinopOp::IsCommutative(kind) && matcher.Is<ConstantOp>(left) &&
        !matcher.Is<ConstantOp>(right)) {
      return ReduceWordBinop(right, left, kind, rep);
    }
    // constant folding
    if (uint64_t k1, k2; matcher.MatchIntegralWordConstant(left, rep, &k1) &&
                         matcher.MatchIntegralWordConstant(right, rep, &k2)) {
      switch (kind) {
        case Kind::kAdd:
          return __ WordConstant(k1 + k2, rep);
        case Kind::kMul:
          return __ WordConstant(k1 * k2, rep);
        case Kind::kBitwiseAnd:
          return __ WordConstant(k1 & k2, rep);
        case Kind::kBitwiseOr:
          return __ WordConstant(k1 | k2, rep);
        case Kind::kBitwiseXor:
          return __ WordConstant(k1 ^ k2, rep);
        case Kind::kSub:
          return __ WordConstant(k1 - k2, rep);
        case Kind::kSignedMulOverflownBits:
          return __ WordConstant(
              is_64 ? base::bits::SignedMulHigh64(static_cast<int64_t>(k1),
                                                  static_cast<int64_t>(k2))
                    : base::bits::SignedMulHigh32(static_cast<int32_t>(k1),
                                                  static_cast<int32_t>(k2)),
              rep);
        case Kind::kUnsignedMulOverflownBits:
          return __ WordConstant(
              is_64 ? base::bits::UnsignedMulHigh64(k1, k2)
                    : base::bits::UnsignedMulHigh32(static_cast<uint32_t>(k1),
                                                    static_cast<uint32_t>(k2)),
              rep);
        case Kind::kSignedDiv:
          return __ WordConstant(
              is_64 ? base::bits::SignedDiv64(k1, k2)
                    : base::bits::SignedDiv32(static_cast<int32_t>(k1),
                                              static_cast<int32_t>(k2)),
              rep);
        case Kind::kUnsignedDiv:
          return __ WordConstant(
              is_64 ? base::bits::UnsignedDiv64(k1, k2)
                    : base::bits::UnsignedDiv32(static_cast<uint32_t>(k1),
                                                static_cast<uint32_t>(k2)),
              rep);
        case Kind::kSignedMod:
          return __ WordConstant(
              is_64 ? base::bits::SignedMod64(k1, k2)
                    : base::bits::SignedMod32(static_cast<int32_t>(k1),
                                              static_cast<int32_t>(k2)),
              rep);
        case Kind::kUnsignedMod:
          return __ WordConstant(
              is_64 ? base::bits::UnsignedMod64(k1, k2)
                    : base::bits::UnsignedMod32(static_cast<uint32_t>(k1),
                                                static_cast<uint32_t>(k2)),
              rep);
      }
    }

    // TODO(tebbi): Detect and merge multiple bitfield checks for CSA/Torque
    // code.

    if (uint64_t right_value;
        matcher.MatchIntegralWordConstant(right, rep, &right_value)) {
      // TODO(jkummerow): computing {right_value_signed} could probably be
      // handled by the 4th argument to {MatchIntegralWordConstant}.
      int64_t right_value_signed =
          is_64 ? static_cast<int64_t>(right_value)
                : int64_t{static_cast<int32_t>(right_value)};
      // (a <op> k1) <op> k2  =>  a <op> (k1 <op> k2)
      if (OpIndex a, k1; WordBinopOp::IsAssociative(kind) &&
                         matcher.MatchWordBinop(left, &a, &k1, kind, rep) &&
                         matcher.Is<ConstantOp>(k1)) {
        OpIndex k2 = right;
        // This optimization allows to do constant folding of `k1` and `k2`.
        // However, if (a <op> k1) has to be calculated anyways, then constant
        // folding does not save any calculations during runtime, and it may
        // increase register pressure because it extends the lifetime of `a`.
        // Therefore we do the optimization only when `left = (a <op k1)` has no
        // other uses.
        if (matcher.Get(left).saturated_use_count.IsZero()) {
          return ReduceWordBinop(a, ReduceWordBinop(k1, k2, kind, rep), kind,
                                 rep);
        }
      }
      switch (kind) {
        case Kind::kSub:
          // left - k  => left + -k
          return ReduceWordBinop(left, __ WordConstant(-right_value, rep),
                                 Kind::kAdd, rep);
        case Kind::kAdd:
          // left + 0  =>  left
          if (right_value == 0) {
            return left;
          }
          break;
        case Kind::kBitwiseXor:
          // left ^ 0  =>  left
          if (right_value == 0) {
            return left;
          }
          // left ^ 1  =>  left == 0  if left is 0 or 1
          if (right_value == 1 && IsBit(left)) {
            return __ Word32Equal(left, __ Word32Constant(0));
          }
          // (x ^ -1) ^ -1  =>  x
          {
            OpIndex x, y;
            int64_t k;
            if (right_value_signed == -1 &&
                matcher.MatchBitwiseAnd(left, &x, &y, rep) &&
                matcher.MatchIntegralWordConstant(y, rep, &k) && k == -1) {
              return x;
            }
          }
          break;
        case Kind::kBitwiseOr:
          // left | 0  =>  left
          if (right_value == 0) {
            return left;
          }
          // left | -1  =>  -1
          if (right_value_signed == -1) {
            return right;
          }
          // (x & K1) | K2 => x | K2 if K2 has ones for every zero bit in K1.
          // This case can be constructed by UpdateWord and UpdateWord32 in CSA.
          {
            OpIndex x, y;
            uint64_t k1;
            uint64_t k2 = right_value;
            if (matcher.MatchBitwiseAnd(left, &x, &y, rep) &&
                matcher.MatchIntegralWordConstant(y, rep, &k1) &&
                (k1 | k2) == rep.MaxUnsignedValue()) {
              return __ WordBitwiseOr(x, right, rep);
            }
          }
          break;
        case Kind::kMul:
          // left * 0  =>  0
          if (right_value == 0) {
            return __ WordConstant(0, rep);
          }
          // left * 1  =>  left
          if (right_value == 1) {
            return left;
          }
          // left * -1 => 0 - left
          if (right_value_signed == -1) {
            return __ WordSub(__ WordConstant(0, rep), left, rep);
          }
          // left * 2^k  =>  left << k
          if (base::bits::IsPowerOfTwo(right_value)) {
            OpIndex shift_amount =
                __ Word32Constant(base::bits::WhichPowerOfTwo(right_value));
            return __ ShiftLeft(left, shift_amount, rep);
          }
          break;
        case Kind::kBitwiseAnd:
          // left & -1 => left
          if (right_value_signed == -1) {
            return left;
          }
          // x & 0  =>  0
          if (right_value == 0) {
            return right;
          }

          if (right_value == 1) {
            // (x + x) & 1  =>  0
            OpIndex left_ignore_extensions =
                IsWord32ConvertedToWord64(left)
                    ? UndoWord32ToWord64Conversion(left)
                    : left;
            if (OpIndex a, b;
                matcher.MatchWordAdd(left_ignore_extensions, &a, &b,
                                     WordRepresentation::Word32()) &&
                a == b) {
              return __ WordConstant(0, rep);
            }

            // CMP & 1  =>  CMP
            if (IsBit(left_ignore_extensions)) {
              return left;
            }

            // HeapObject & 1 => 1  ("& 1" is a Smi-check)
            // Note that we don't constant-fold the general case of
            // "HeapObject binop cst", because it's a bit unclear when such
            // operations would be used outside of smi-checks, and it's thus
            // unclear whether constant-folding would be safe.
            if (const ConstantOp* cst = matcher.TryCast<ConstantOp>(left)) {
              if (cst->kind ==
                  any_of(ConstantOp::Kind::kHeapObject,
                         ConstantOp::Kind::kCompressedHeapObject)) {
                return __ WordConstant(1, rep);
              }
            }
          }

          // asm.js often benefits from these transformations, to optimize out
          // unnecessary memory access alignment masks. Conventions used in
          // the comments below:
          // x, y: arbitrary values
          // K, L, M: arbitrary constants
          // (-1 << K) == mask: the right-hand side of the bitwise AND.
          if (IsNegativePowerOfTwo(right_value_signed)) {
            uint64_t mask = right_value;
            int K = base::bits::CountTrailingZeros64(mask);
            OpIndex x, y;
            {
              int L;
              //   (x << L) & (-1 << K)
              // => x << L               iff L >= K
              if (matcher.MatchConstantLeftShift(left, &x, rep, &L) && L >= K) {
                return left;
              }
            }

            if (matcher.MatchWordAdd(left, &x, &y, rep)) {
              uint64_t L;  // L == (M << K) iff (L & mask) == L.

              //    (x              + (M << K)) & (-1 << K)
              // => (x & (-1 << K)) + (M << K)
              if (matcher.MatchIntegralWordConstant(y, rep, &L) &&
                  (L & mask) == L) {
                return __ WordAdd(__ WordBitwiseAnd(x, right, rep),
                                  __ WordConstant(L, rep), rep);
              }

              //   (x1 * (M << K) + y) & (-1 << K)
              // => x1 * (M << K) + (y & (-1 << K))
              OpIndex x1, x2, y1, y2;
              if (matcher.MatchWordMul(x, &x1, &x2, rep) &&
                  matcher.MatchIntegralWordConstant(x2, rep, &L) &&
                  (L & mask) == L) {
                return __ WordAdd(x, __ WordBitwiseAnd(y, right, rep), rep);
              }
              // Same as above with swapped order:
              //    (x              + y1 * (M << K)) & (-1 << K)
              // => (x & (-1 << K)) + y1 * (M << K)
              if (matcher.MatchWordMul(y, &y1, &y2, rep) &&
                  matcher.MatchIntegralWordConstant(y2, rep, &L) &&
                  (L & mask) == L) {
                return __ WordAdd(__ WordBitwiseAnd(x, right, rep), y, rep);
              }

              //   ((x1 << K) + y) & (-1 << K)
              // => (x1 << K) + (y & (-1 << K))
              int K2;
              if (matcher.MatchConstantLeftShift(x, &x1, rep, &K2) && K2 == K) {
                return __ WordAdd(x, __ WordBitwiseAnd(y, right, rep), rep);
              }
              // Same as above with swapped order:
              //    (x +              (y1 << K)) & (-1 << K)
              // => (x & (-1 << K)) + (y1 << K)
              if (matcher.MatchConstantLeftShift(y, &y1, rep, &K2) && K2 == K) {
                return __ WordAdd(__ WordBitwiseAnd(x, right, rep), y, rep);
              }
            }
          }
          break;
        case WordBinopOp::Kind::kSignedDiv:
          return ReduceSignedDiv(left, right_value_signed, rep);
        case WordBinopOp::Kind::kUnsignedDiv:
          return ReduceUnsignedDiv(left, right_value, rep);
        case WordBinopOp::Kind::kSignedMod:
          // left % 0  =>  0
          // left % 1  =>  0
          // left % -1  =>  0
          if (right_value_signed == any_of(0, 1, -1)) {
            return __ WordConstant(0, rep);
          }
          if (right_value_signed != rep.MinSignedValue()) {
            right_value_signed = Abs(right_value_signed);
          }
          // left % 2^n  =>  ((left + m) & (2^n - 1)) - m
          // where m = (left >> bits-1) >>> bits-n
          // This is a branch-free version of the following:
          // left >= 0 ? left & (2^n - 1)
          //           : ((left + (2^n - 1)) & (2^n - 1)) - (2^n - 1)
          // Adding and subtracting (2^n - 1) before and after the bitwise-and
          // keeps the result congruent modulo 2^n, but shifts the resulting
          // value range to become -(2^n - 1) ... 0.
          if (base::bits::IsPowerOfTwo(right_value_signed)) {
            uint32_t bits = rep.bit_width();
            uint32_t n = base::bits::WhichPowerOfTwo(right_value_signed);
            OpIndex m = __ ShiftRightLogical(
                __ ShiftRightArithmetic(left, bits - 1, rep), bits - n, rep);
            return __ WordSub(
                __ WordBitwiseAnd(__ WordAdd(left, m, rep),
                                  __ WordConstant(right_value_signed - 1, rep),
                                  rep),
                m, rep);
          }
          // The `IntDiv` with a constant right-hand side will be turned into a
          // multiplication, avoiding the expensive integer division.
          return __ WordSub(
              left, __ WordMul(__ IntDiv(left, right, rep), right, rep), rep);
        case WordBinopOp::Kind::kUnsignedMod:
          // left % 0  =>  0
          // left % 1  =>  0
          if (right_value == 0 || right_value == 1) {
            return __ WordConstant(0, rep);
          }
          // x % 2^n => x & (2^n - 1)
          if (base::bits::IsPowerOfTwo(right_value)) {
            return __ WordBitwiseAnd(
                left, __ WordConstant(right_value - 1, rep), rep);
          }
          // The `UintDiv` with a constant right-hand side will be turned into a
          // multiplication, avoiding the expensive integer division.
          return __ WordSub(
              left, __ WordMul(right, __ UintDiv(left, right, rep), rep), rep);
        case WordBinopOp::Kind::kSignedMulOverflownBits:
        case WordBinopOp::Kind::kUnsignedMulOverflownBits:
          break;
      }
    }

    if (kind == Kind::kAdd) {
      OpIndex x, y, zero;
      // (0 - x) + y => y - x
      if (matcher.MatchWordSub(left, &zero, &x, rep) &&
          matcher.MatchZero(zero)) {
        y = right;
        return __ WordSub(y, x, rep);
      }
      // x + (0 - y) => x - y
      if (matcher.MatchWordSub(right, &zero, &y, rep) &&
          matcher.MatchZero(zero)) {
        x = left;
        return __ WordSub(x, y, rep);
      }
    }

    // 0 / right  =>  0
    // 0 % right  =>  0
    if (matcher.MatchZero(left) &&
        kind == any_of(Kind::kSignedDiv, Kind::kUnsignedDiv, Kind::kUnsignedMod,
                       Kind::kSignedMod)) {
      return __ WordConstant(0, rep);
    }

    if (left == right) {
      OpIndex x = left;
      switch (kind) {
        // x & x  =>  x
        // x | x  =>  x
        case WordBinopOp::Kind::kBitwiseAnd:
        case WordBinopOp::Kind::kBitwiseOr:
          return x;
        // x ^ x  =>  0
        // x - x  =>  0
        // x % x  =>  0
        case WordBinopOp::Kind::kBitwiseXor:
        case WordBinopOp::Kind::kSub:
        case WordBinopOp::Kind::kSignedMod:
        case WordBinopOp::Kind::kUnsignedMod:
          return __ WordConstant(0, rep);
        // x / x  =>  x != 0
        case WordBinopOp::Kind::kSignedDiv:
        case WordBinopOp::Kind::kUnsignedDiv: {
          OpIndex zero = __ WordConstant(0, rep);
          V<Word32> result = __ Word32Equal(__ Equal(left, zero, rep), 0);
          return __ ZeroExtendWord32ToRep(result, rep);
        }
        case WordBinopOp::Kind::kAdd:
        case WordBinopOp::Kind::kMul:
        case WordBinopOp::Kind::kSignedMulOverflownBits:
        case WordBinopOp::Kind::kUnsignedMulOverflownBits:
          break;
      }
    }

    if (base::Optional<OpIndex> ror = TryReduceToRor(left, right, kind, rep)) {
      return *ror;
    }

    return Next::ReduceWordBinop(left, right, kind, rep);
  }
