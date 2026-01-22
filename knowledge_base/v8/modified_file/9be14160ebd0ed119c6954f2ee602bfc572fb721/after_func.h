  OpIndex REDUCE(OverflowCheckedBinop)(OpIndex left, OpIndex right,
                                       OverflowCheckedBinopOp::Kind kind,
                                       WordRepresentation rep) {
    if (ShouldSkipOptimizationStep()) {
      return Next::ReduceOverflowCheckedBinop(left, right, kind, rep);
    }
    using Kind = OverflowCheckedBinopOp::Kind;
    if (OverflowCheckedBinopOp::IsCommutative(kind) &&
        matcher.Is<ConstantOp>(left) && !matcher.Is<ConstantOp>(right)) {
      return ReduceOverflowCheckedBinop(right, left, kind, rep);
    }
    if (rep == WordRepresentation::Word32()) {
      left = TryRemoveWord32ToWord64Conversion(left);
      right = TryRemoveWord32ToWord64Conversion(right);
    }
    // constant folding
    if (rep == WordRepresentation::Word32()) {
      if (int32_t k1, k2; matcher.MatchIntegralWord32Constant(left, &k1) &&
                          matcher.MatchIntegralWord32Constant(right, &k2)) {
        bool overflow;
        int32_t res;
        switch (kind) {
          case OverflowCheckedBinopOp::Kind::kSignedAdd:
            overflow = base::bits::SignedAddOverflow32(k1, k2, &res);
            break;
          case OverflowCheckedBinopOp::Kind::kSignedMul:
            overflow = base::bits::SignedMulOverflow32(k1, k2, &res);
            break;
          case OverflowCheckedBinopOp::Kind::kSignedSub:
            overflow = base::bits::SignedSubOverflow32(k1, k2, &res);
            break;
        }
        return __ Tuple(__ Word32Constant(res), __ Word32Constant(overflow));
      }
    } else {
      DCHECK_EQ(rep, WordRepresentation::Word64());
      if (int64_t k1, k2; matcher.MatchIntegralWord64Constant(left, &k1) &&
                          matcher.MatchIntegralWord64Constant(right, &k2)) {
        bool overflow;
        int64_t res;
        switch (kind) {
          case OverflowCheckedBinopOp::Kind::kSignedAdd:
            overflow = base::bits::SignedAddOverflow64(k1, k2, &res);
            break;
          case OverflowCheckedBinopOp::Kind::kSignedMul:
            overflow = base::bits::SignedMulOverflow64(k1, k2, &res);
            break;
          case OverflowCheckedBinopOp::Kind::kSignedSub:
            overflow = base::bits::SignedSubOverflow64(k1, k2, &res);
            break;
        }
        return __ Tuple(__ Word64Constant(res), __ Word32Constant(overflow));
      }
    }

    // left + 0  =>  (left, false)
    // left - 0  =>  (left, false)
    if (kind == any_of(Kind::kSignedAdd, Kind::kSignedSub) &&
        matcher.MatchZero(right)) {
      return __ Tuple(left, right);
    }

    if (kind == Kind::kSignedMul) {
      if (int64_t k; matcher.MatchIntegralWordConstant(right, rep, &k)) {
        // left * 0  =>  (0, false)
        if (k == 0) {
          return __ Tuple(__ WordConstant(0, rep), __ Word32Constant(false));
        }
        // left * 1  =>  (left, false)
        if (k == 1) {
          return __ Tuple(left, __ Word32Constant(false));
        }
        // left * -1  =>  0 - left
        if (k == -1) {
          return __ IntSubCheckOverflow(__ WordConstant(0, rep), left, rep);
        }
        // left * 2  =>  left + left
        if (k == 2) {
          return __ IntAddCheckOverflow(left, left, rep);
        }
      }
    }

    // UntagSmi(x) + UntagSmi(x)  =>  (x, false)
    // (where UntagSmi(x) = x >> 1   with a ShiftOutZeros shift)
    if (kind == Kind::kSignedAdd && left == right) {
      uint16_t amount;
      if (OpIndex x; matcher.MatchConstantShiftRightArithmeticShiftOutZeros(
                         left, &x, WordRepresentation::Word32(), &amount) &&
                     amount == 1) {
        return __ Tuple(x, __ Word32Constant(0));
      }
    }

    return Next::ReduceOverflowCheckedBinop(left, right, kind, rep);
  }
