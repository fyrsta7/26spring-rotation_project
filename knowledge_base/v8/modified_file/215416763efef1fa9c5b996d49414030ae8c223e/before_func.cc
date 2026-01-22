Reduction MachineOperatorReducer::Reduce(Node* node) {
  switch (node->opcode()) {
    case IrOpcode::kProjection:
      return ReduceProjection(ProjectionIndexOf(node->op()), node->InputAt(0));
    case IrOpcode::kWord32And:
      return ReduceWord32And(node);
    case IrOpcode::kWord64And:
      return ReduceWord64And(node);
    case IrOpcode::kWord32Or:
      return ReduceWord32Or(node);
    case IrOpcode::kWord64Or:
      return ReduceWord64Or(node);
    case IrOpcode::kWord32Xor:
      return ReduceWord32Xor(node);
    case IrOpcode::kWord64Xor:
      return ReduceWord64Xor(node);
    case IrOpcode::kWord32Shl:
      return ReduceWord32Shl(node);
    case IrOpcode::kWord64Shl:
      return ReduceWord64Shl(node);
    case IrOpcode::kWord32Shr:
      return ReduceWord32Shr(node);
    case IrOpcode::kWord64Shr:
      return ReduceWord64Shr(node);
    case IrOpcode::kWord32Sar:
      return ReduceWord32Sar(node);
    case IrOpcode::kWord64Sar:
      return ReduceWord64Sar(node);
    case IrOpcode::kWord32Ror: {
      Int32BinopMatcher m(node);
      if (m.right().Is(0)) return Replace(m.left().node());  // x ror 0 => x
      if (m.IsFoldable()) {  // K ror K => K  (K stands for arbitrary constants)
        return ReplaceInt32(base::bits::RotateRight32(
            m.left().ResolvedValue(), m.right().ResolvedValue() & 31));
      }
      break;
    }
    case IrOpcode::kWord32Equal: {
      return ReduceWord32Equal(node);
    }
    case IrOpcode::kWord64Equal: {
      Int64BinopMatcher m(node);
      if (m.IsFoldable()) {  // K == K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() ==
                           m.right().ResolvedValue());
      }
      if (m.left().IsInt64Sub() && m.right().Is(0)) {  // x - y == 0 => x == y
        Int64BinopMatcher msub(m.left().node());
        node->ReplaceInput(0, msub.left().node());
        node->ReplaceInput(1, msub.right().node());
        return Changed(node);
      }
      // TODO(turbofan): fold HeapConstant, ExternalReference, pointer compares
      if (m.LeftEqualsRight()) return ReplaceBool(true);  // x == x => true
      break;
    }
    case IrOpcode::kInt32Add:
      return ReduceInt32Add(node);
    case IrOpcode::kInt64Add:
      return ReduceInt64Add(node);
    case IrOpcode::kInt32Sub:
      return ReduceInt32Sub(node);
    case IrOpcode::kInt64Sub:
      return ReduceInt64Sub(node);
    case IrOpcode::kInt32Mul: {
      Int32BinopMatcher m(node);
      if (m.right().Is(0)) return Replace(m.right().node());  // x * 0 => 0
      if (m.right().Is(1)) return Replace(m.left().node());   // x * 1 => x
      if (m.IsFoldable()) {  // K * K => K  (K stands for arbitrary constants)
        return ReplaceInt32(base::MulWithWraparound(m.left().ResolvedValue(),
                                                    m.right().ResolvedValue()));
      }
      if (m.right().Is(-1)) {  // x * -1 => 0 - x
        node->ReplaceInput(0, Int32Constant(0));
        node->ReplaceInput(1, m.left().node());
        NodeProperties::ChangeOp(node, machine()->Int32Sub());
        return Changed(node);
      }
      if (m.right().IsPowerOf2()) {  // x * 2^n => x << n
        node->ReplaceInput(1, Int32Constant(base::bits::WhichPowerOfTwo(
                                  m.right().ResolvedValue())));
        NodeProperties::ChangeOp(node, machine()->Word32Shl());
        return Changed(node).FollowedBy(ReduceWord32Shl(node));
      }
      // (x * Int32Constant(a)) * Int32Constant(b)) => x * Int32Constant(a * b)
      if (m.right().HasResolvedValue() && m.left().IsInt32Mul()) {
        Int32BinopMatcher n(m.left().node());
        if (n.right().HasResolvedValue() && m.OwnsInput(m.left().node())) {
          node->ReplaceInput(
              1, Int32Constant(base::MulWithWraparound(
                     m.right().ResolvedValue(), n.right().ResolvedValue())));
          node->ReplaceInput(0, n.left().node());
          return Changed(node);
        }
      }
      break;
    }
    case IrOpcode::kInt32MulWithOverflow: {
      Int32BinopMatcher m(node);
      if (m.right().Is(2)) {
        node->ReplaceInput(1, m.left().node());
        NodeProperties::ChangeOp(node, machine()->Int32AddWithOverflow());
        return Changed(node);
      }
      if (m.right().Is(-1)) {
        node->ReplaceInput(0, Int32Constant(0));
        node->ReplaceInput(1, m.left().node());
        NodeProperties::ChangeOp(node, machine()->Int32SubWithOverflow());
        return Changed(node);
      }
      break;
    }
    case IrOpcode::kInt64Mul:
      return ReduceInt64Mul(node);
    case IrOpcode::kInt32Div:
      return ReduceInt32Div(node);
    case IrOpcode::kUint32Div:
      return ReduceUint32Div(node);
    case IrOpcode::kInt32Mod:
      return ReduceInt32Mod(node);
    case IrOpcode::kUint32Mod:
      return ReduceUint32Mod(node);
    case IrOpcode::kInt32LessThan: {
      Int32BinopMatcher m(node);
      if (m.IsFoldable()) {  // K < K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() <
                           m.right().ResolvedValue());
      }
      if (m.LeftEqualsRight()) return ReplaceBool(false);  // x < x => false
      if (m.left().IsWord32Or() && m.right().Is(0)) {
        // (x | K) < 0 => true or (K | x) < 0 => true iff K < 0
        Int32BinopMatcher mleftmatcher(m.left().node());
        if (mleftmatcher.left().IsNegative() ||
            mleftmatcher.right().IsNegative()) {
          return ReplaceBool(true);
        }
      }
      return ReduceWord32Comparisons(node);
    }
    case IrOpcode::kInt32LessThanOrEqual: {
      Int32BinopMatcher m(node);
      if (m.IsFoldable()) {  // K <= K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() <=
                           m.right().ResolvedValue());
      }
      if (m.LeftEqualsRight()) return ReplaceBool(true);  // x <= x => true
      return ReduceWord32Comparisons(node);
    }
    case IrOpcode::kUint32LessThan: {
      Uint32BinopMatcher m(node);
      if (m.left().Is(kMaxUInt32)) return ReplaceBool(false);  // M < x => false
      if (m.right().Is(0)) return ReplaceBool(false);          // x < 0 => false
      if (m.IsFoldable()) {  // K < K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() <
                           m.right().ResolvedValue());
      }
      if (m.LeftEqualsRight()) return ReplaceBool(false);  // x < x => false
      if (m.left().IsWord32Sar() && m.right().HasResolvedValue()) {
        Int32BinopMatcher mleft(m.left().node());
        if (mleft.right().HasResolvedValue()) {
          // (x >> K) < C => x < (C << K)
          // when C < (M >> K)
          const uint32_t c = m.right().ResolvedValue();
          const uint32_t k = mleft.right().ResolvedValue() & 0x1F;
          if (c < static_cast<uint32_t>(kMaxInt >> k)) {
            node->ReplaceInput(0, mleft.left().node());
            node->ReplaceInput(1, Uint32Constant(c << k));
            return Changed(node);
          }
          // TODO(turbofan): else the comparison is always true.
        }
      }
      return ReduceWord32Comparisons(node);
    }
    case IrOpcode::kUint32LessThanOrEqual: {
      Uint32BinopMatcher m(node);
      if (m.left().Is(0)) return ReplaceBool(true);            // 0 <= x => true
      if (m.right().Is(kMaxUInt32)) return ReplaceBool(true);  // x <= M => true
      if (m.IsFoldable()) {  // K <= K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() <=
                           m.right().ResolvedValue());
      }
      if (m.LeftEqualsRight()) return ReplaceBool(true);  // x <= x => true
      return ReduceWord32Comparisons(node);
    }
    case IrOpcode::kFloat32Sub: {
      Float32BinopMatcher m(node);
      if (allow_signalling_nan_ && m.right().Is(0) &&
          (std::copysign(1.0, m.right().ResolvedValue()) > 0)) {
        return Replace(m.left().node());  // x - 0 => x
      }
      if (m.right().IsNaN()) {  // x - NaN => NaN
        return ReplaceFloat32(SilenceNaN(m.right().ResolvedValue()));
      }
      if (m.left().IsNaN()) {  // NaN - x => NaN
        return ReplaceFloat32(SilenceNaN(m.left().ResolvedValue()));
      }
      if (m.IsFoldable()) {  // L - R => (L - R)
        return ReplaceFloat32(m.left().ResolvedValue() -
                              m.right().ResolvedValue());
      }
      if (allow_signalling_nan_ && m.left().IsMinusZero()) {
        // -0.0 - round_down(-0.0 - R) => round_up(R)
        if (machine()->Float32RoundUp().IsSupported() &&
            m.right().IsFloat32RoundDown()) {
          if (m.right().InputAt(0)->opcode() == IrOpcode::kFloat32Sub) {
            Float32BinopMatcher mright0(m.right().InputAt(0));
            if (mright0.left().IsMinusZero()) {
              return Replace(graph()->NewNode(machine()->Float32RoundUp().op(),
                                              mright0.right().node()));
            }
          }
        }
        // -0.0 - R => -R
        node->RemoveInput(0);
        NodeProperties::ChangeOp(node, machine()->Float32Neg());
        return Changed(node);
      }
      break;
    }
    case IrOpcode::kFloat64Add: {
      Float64BinopMatcher m(node);
      if (m.IsFoldable()) {  // K + K => K  (K stands for arbitrary constants)
        return ReplaceFloat64(m.left().ResolvedValue() +
                              m.right().ResolvedValue());
      }
      break;
    }
    case IrOpcode::kFloat64Sub: {
      Float64BinopMatcher m(node);
      if (allow_signalling_nan_ && m.right().Is(0) &&
          (Double(m.right().ResolvedValue()).Sign() > 0)) {
        return Replace(m.left().node());  // x - 0 => x
      }
      if (m.right().IsNaN()) {  // x - NaN => NaN
        return ReplaceFloat64(SilenceNaN(m.right().ResolvedValue()));
      }
      if (m.left().IsNaN()) {  // NaN - x => NaN
        return ReplaceFloat64(SilenceNaN(m.left().ResolvedValue()));
      }
      if (m.IsFoldable()) {  // L - R => (L - R)
        return ReplaceFloat64(m.left().ResolvedValue() -
                              m.right().ResolvedValue());
      }
      if (allow_signalling_nan_ && m.left().IsMinusZero()) {
        // -0.0 - round_down(-0.0 - R) => round_up(R)
        if (machine()->Float64RoundUp().IsSupported() &&
            m.right().IsFloat64RoundDown()) {
          if (m.right().InputAt(0)->opcode() == IrOpcode::kFloat64Sub) {
            Float64BinopMatcher mright0(m.right().InputAt(0));
            if (mright0.left().IsMinusZero()) {
              return Replace(graph()->NewNode(machine()->Float64RoundUp().op(),
                                              mright0.right().node()));
            }
          }
        }
        // -0.0 - R => -R
        node->RemoveInput(0);
        NodeProperties::ChangeOp(node, machine()->Float64Neg());
        return Changed(node);
      }
      break;
    }
    case IrOpcode::kFloat64Mul: {
      Float64BinopMatcher m(node);
      if (allow_signalling_nan_ && m.right().Is(1))
        return Replace(m.left().node());  // x * 1.0 => x
      if (m.right().Is(-1)) {  // x * -1.0 => -0.0 - x
        node->ReplaceInput(0, Float64Constant(-0.0));
        node->ReplaceInput(1, m.left().node());
        NodeProperties::ChangeOp(node, machine()->Float64Sub());
        return Changed(node);
      }
      if (m.right().IsNaN()) {                               // x * NaN => NaN
        return ReplaceFloat64(SilenceNaN(m.right().ResolvedValue()));
      }
      if (m.IsFoldable()) {  // K * K => K  (K stands for arbitrary constants)
        return ReplaceFloat64(m.left().ResolvedValue() *
                              m.right().ResolvedValue());
      }
      if (m.right().Is(2)) {  // x * 2.0 => x + x
        node->ReplaceInput(1, m.left().node());
        NodeProperties::ChangeOp(node, machine()->Float64Add());
        return Changed(node);
      }
      break;
    }
    case IrOpcode::kFloat64Div: {
      Float64BinopMatcher m(node);
      if (allow_signalling_nan_ && m.right().Is(1))
        return Replace(m.left().node());  // x / 1.0 => x
      // TODO(ahaas): We could do x / 1.0 = x if we knew that x is not an sNaN.
      if (m.right().IsNaN()) {                               // x / NaN => NaN
        return ReplaceFloat64(SilenceNaN(m.right().ResolvedValue()));
      }
      if (m.left().IsNaN()) {  // NaN / x => NaN
        return ReplaceFloat64(SilenceNaN(m.left().ResolvedValue()));
      }
      if (m.IsFoldable()) {  // K / K => K  (K stands for arbitrary constants)
        return ReplaceFloat64(
            base::Divide(m.left().ResolvedValue(), m.right().ResolvedValue()));
      }
      if (allow_signalling_nan_ && m.right().Is(-1)) {  // x / -1.0 => -x
        node->RemoveInput(1);
        NodeProperties::ChangeOp(node, machine()->Float64Neg());
        return Changed(node);
      }
      if (m.right().IsNormal() && m.right().IsPositiveOrNegativePowerOf2()) {
        // All reciprocals of non-denormal powers of two can be represented
        // exactly, so division by power of two can be reduced to
        // multiplication by reciprocal, with the same result.
        node->ReplaceInput(1, Float64Constant(1.0 / m.right().ResolvedValue()));
        NodeProperties::ChangeOp(node, machine()->Float64Mul());
        return Changed(node);
      }
      break;
    }
    case IrOpcode::kFloat64Mod: {
      Float64BinopMatcher m(node);
      if (m.right().Is(0)) {  // x % 0 => NaN
        return ReplaceFloat64(std::numeric_limits<double>::quiet_NaN());
      }
      if (m.right().IsNaN()) {  // x % NaN => NaN
        return Replace(m.right().node());
      }
      if (m.left().IsNaN()) {  // NaN % x => NaN
        return Replace(m.left().node());
      }
      if (m.IsFoldable()) {  // K % K => K  (K stands for arbitrary constants)
        return ReplaceFloat64(
            Modulo(m.left().ResolvedValue(), m.right().ResolvedValue()));
      }
      break;
    }
    case IrOpcode::kFloat64Acos: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::acos(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Acosh: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::acosh(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Asin: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::asin(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Asinh: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::asinh(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Atan: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::atan(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Atanh: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::atanh(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Atan2: {
      Float64BinopMatcher m(node);
      if (m.right().IsNaN()) {
        return Replace(m.right().node());
      }
      if (m.left().IsNaN()) {
        return Replace(m.left().node());
      }
      if (m.IsFoldable()) {
        return ReplaceFloat64(base::ieee754::atan2(m.left().ResolvedValue(),
                                                   m.right().ResolvedValue()));
      }
      break;
    }
    case IrOpcode::kFloat64Cbrt: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::cbrt(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Cos: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::cos(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Cosh: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::cosh(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Exp: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::exp(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Expm1: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::expm1(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Log: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::log(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Log1p: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::log1p(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Log10: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::log10(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Log2: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::log2(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Pow: {
      Float64BinopMatcher m(node);
      if (m.IsFoldable()) {
        return ReplaceFloat64(base::ieee754::pow(m.left().ResolvedValue(),
                                                 m.right().ResolvedValue()));
      } else if (m.right().Is(0.0)) {  // x ** +-0.0 => 1.0
        return ReplaceFloat64(1.0);
      } else if (m.right().Is(-2.0)) {  // x ** -2.0 => 1 / (x * x)
        node->ReplaceInput(0, Float64Constant(1.0));
        node->ReplaceInput(1, Float64Mul(m.left().node(), m.left().node()));
        NodeProperties::ChangeOp(node, machine()->Float64Div());
        return Changed(node);
      } else if (m.right().Is(2.0)) {  // x ** 2.0 => x * x
        node->ReplaceInput(1, m.left().node());
        NodeProperties::ChangeOp(node, machine()->Float64Mul());
        return Changed(node);
      } else if (m.right().Is(-0.5)) {
        // x ** 0.5 => 1 / (if x <= -Infinity then Infinity else sqrt(0.0 + x))
        node->ReplaceInput(0, Float64Constant(1.0));
        node->ReplaceInput(1, Float64PowHalf(m.left().node()));
        NodeProperties::ChangeOp(node, machine()->Float64Div());
        return Changed(node);
      } else if (m.right().Is(0.5)) {
        // x ** 0.5 => if x <= -Infinity then Infinity else sqrt(0.0 + x)
        return Replace(Float64PowHalf(m.left().node()));
      }
      break;
    }
    case IrOpcode::kFloat64Sin: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::sin(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Sinh: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::sinh(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Tan: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::tan(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kFloat64Tanh: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(base::ieee754::tanh(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kChangeFloat32ToFloat64: {
      Float32Matcher m(node->InputAt(0));
      if (m.HasResolvedValue()) {
        if (!allow_signalling_nan_ && std::isnan(m.ResolvedValue())) {
          return ReplaceFloat64(SilenceNaN(m.ResolvedValue()));
        }
        return ReplaceFloat64(m.ResolvedValue());
      }
      break;
    }
    case IrOpcode::kChangeFloat64ToInt32: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceInt32(FastD2IChecked(m.ResolvedValue()));
      if (m.IsChangeInt32ToFloat64()) return Replace(m.node()->InputAt(0));
      break;
    }
    case IrOpcode::kChangeFloat64ToInt64: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceInt64(static_cast<int64_t>(m.ResolvedValue()));
      if (m.IsChangeInt64ToFloat64()) return Replace(m.node()->InputAt(0));
      break;
    }
    case IrOpcode::kChangeFloat64ToUint32: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceInt32(FastD2UI(m.ResolvedValue()));
      if (m.IsChangeUint32ToFloat64()) return Replace(m.node()->InputAt(0));
      break;
    }
    case IrOpcode::kChangeInt32ToFloat64: {
      Int32Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(FastI2D(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kBitcastWord32ToWord64: {
      Int32Matcher m(node->InputAt(0));
      if (m.HasResolvedValue()) return ReplaceInt64(m.ResolvedValue());
      break;
    }
    case IrOpcode::kChangeInt32ToInt64: {
      Int32Matcher m(node->InputAt(0));
      if (m.HasResolvedValue()) return ReplaceInt64(m.ResolvedValue());
      break;
    }
    case IrOpcode::kChangeInt64ToFloat64: {
      Int64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(static_cast<double>(m.ResolvedValue()));
      if (m.IsChangeFloat64ToInt64()) return Replace(m.node()->InputAt(0));
      break;
    }
    case IrOpcode::kChangeUint32ToFloat64: {
      Uint32Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceFloat64(FastUI2D(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kChangeUint32ToUint64: {
      Uint32Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceInt64(static_cast<uint64_t>(m.ResolvedValue()));
      break;
    }
    case IrOpcode::kTruncateFloat64ToWord32: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue())
        return ReplaceInt32(DoubleToInt32(m.ResolvedValue()));
      if (m.IsChangeInt32ToFloat64()) return Replace(m.node()->InputAt(0));
      return NoChange();
    }
    case IrOpcode::kTruncateInt64ToInt32:
      return ReduceTruncateInt64ToInt32(node);
    case IrOpcode::kTruncateFloat64ToFloat32: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue()) {
        if (!allow_signalling_nan_ && m.IsNaN()) {
          return ReplaceFloat32(DoubleToFloat32(SilenceNaN(m.ResolvedValue())));
        }
        return ReplaceFloat32(DoubleToFloat32(m.ResolvedValue()));
      }
      if (allow_signalling_nan_ && m.IsChangeFloat32ToFloat64())
        return Replace(m.node()->InputAt(0));
      break;
    }
    case IrOpcode::kRoundFloat64ToInt32: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasResolvedValue()) {
        return ReplaceInt32(DoubleToInt32(m.ResolvedValue()));
      }
      if (m.IsChangeInt32ToFloat64()) return Replace(m.node()->InputAt(0));
      break;
    }
    case IrOpcode::kFloat64InsertLowWord32:
      return ReduceFloat64InsertLowWord32(node);
    case IrOpcode::kFloat64InsertHighWord32:
      return ReduceFloat64InsertHighWord32(node);
    case IrOpcode::kStore:
    case IrOpcode::kUnalignedStore:
      return ReduceStore(node);
    case IrOpcode::kFloat64Equal:
    case IrOpcode::kFloat64LessThan:
    case IrOpcode::kFloat64LessThanOrEqual:
      return ReduceFloat64Compare(node);
    case IrOpcode::kFloat64RoundDown:
      return ReduceFloat64RoundDown(node);
    case IrOpcode::kBitcastTaggedToWord:
    case IrOpcode::kBitcastTaggedToWordForTagAndSmiBits: {
      NodeMatcher m(node->InputAt(0));
      if (m.IsBitcastWordToTaggedSigned()) {
        RelaxEffectsAndControls(node);
        return Replace(m.InputAt(0));
      }
      break;
    }
    case IrOpcode::kBranch:
    case IrOpcode::kDeoptimizeIf:
    case IrOpcode::kDeoptimizeUnless:
    case IrOpcode::kTrapIf:
    case IrOpcode::kTrapUnless:
      return ReduceConditional(node);
    case IrOpcode::kInt64LessThan: {
      Int64BinopMatcher m(node);
      if (m.IsFoldable()) {  // K < K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() <
                           m.right().ResolvedValue());
      }
      return ReduceWord64Comparisons(node);
    }
    case IrOpcode::kInt64LessThanOrEqual: {
      Int64BinopMatcher m(node);
      if (m.IsFoldable()) {  // K <= K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() <=
                           m.right().ResolvedValue());
      }
      return ReduceWord64Comparisons(node);
    }
    case IrOpcode::kUint64LessThan: {
      Uint64BinopMatcher m(node);
      if (m.IsFoldable()) {  // K < K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() <
                           m.right().ResolvedValue());
      }
      return ReduceWord64Comparisons(node);
    }
    case IrOpcode::kUint64LessThanOrEqual: {
      Uint64BinopMatcher m(node);
      if (m.IsFoldable()) {  // K <= K => K  (K stands for arbitrary constants)
        return ReplaceBool(m.left().ResolvedValue() <=
                           m.right().ResolvedValue());
      }
      return ReduceWord64Comparisons(node);
    }
    default:
      break;
  }
  return NoChange();
}
