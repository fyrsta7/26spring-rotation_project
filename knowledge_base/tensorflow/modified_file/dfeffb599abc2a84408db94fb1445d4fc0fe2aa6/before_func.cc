    if (!reduced_dims.contains(operand_dim)) {
      if (FindOrDie(unreduced_dim_map, operand_dim) !=
          result_shape.layout().minor_to_major(result_dim_idx++)) {
        return false;
      }
    }
  }

  CHECK_EQ(result_dim_idx, result_shape.dimensions_size());

  return true;
}

IrEmitter::ReductionGenerator IrEmitter::MatchReductionGenerator(
    HloComputation* function, string* failure_reason) const {
  CHECK_EQ(function->num_parameters(), 2);

  auto root_instruction = function->root_instruction();
  CHECK(ShapeUtil::IsScalar(root_instruction->shape()));

  if (root_instruction->operand_count() != 2) {
    *failure_reason = "root instruction is not a binary operation";
    return nullptr;
  }

  const Shape& root_shape = root_instruction->shape();
  if (ShapeUtil::ElementIsComplex(root_shape)) {
    // TODO(b/65408531): Complex add could by done via bitcast to <float x [2N]>
    // Complex multiply would be more challenging. We could perhaps use a
    // strided load to get all reals in a vector, all images in a vector, or use
    // CreateShuffleVector on a bitcast to float x [2N].
    *failure_reason = "complex values not supported";
    return nullptr;
  }
  bool root_is_floating_point = ShapeUtil::ElementIsFloating(root_shape);
  bool root_is_integral = ShapeUtil::ElementIsIntegral(root_shape);
  bool root_is_signed = ShapeUtil::ElementIsSigned(root_shape);

  auto lhs = root_instruction->operand(0);
  auto rhs = root_instruction->operand(1);

  auto param_0 = function->parameter_instruction(0);
  auto param_1 = function->parameter_instruction(1);
  if (!(lhs == param_0 && rhs == param_1) &&
      !(rhs == param_0 && lhs == param_1)) {
    *failure_reason =
        "root instruction is not a binary operation on the incoming arguments";
    return nullptr;
  }

  CHECK(ShapeUtil::IsScalar(lhs->shape()) && ShapeUtil::IsScalar(rhs->shape()));

  // This is visually similar to ElementalIrEmitter, though conceptually we're
  // doing something different here.  ElementalIrEmitter emits scalar operations
  // while these emit scalar or vector operations depending on the type of the
  // operands. See CreateShardedVectorType for the actual types in use here.
  switch (root_instruction->opcode()) {
    default:
      *failure_reason = "did not recognize root instruction opcode";
      return nullptr;

    case HloOpcode::kAdd:
      return [root_is_integral](llvm::IRBuilder<>* b, llvm::Value* lhs,
                                llvm::Value* rhs) {
        return root_is_integral ? b->CreateAdd(lhs, rhs)
                                : b->CreateFAdd(lhs, rhs);
      };

    case HloOpcode::kMultiply:
      return [root_is_integral](llvm::IRBuilder<>* b, llvm::Value* lhs,
                                llvm::Value* rhs) {
        return root_is_integral ? b->CreateMul(lhs, rhs)
                                : b->CreateFMul(lhs, rhs);
      };

    case HloOpcode::kAnd:
      return [](llvm::IRBuilder<>* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateAnd(lhs, rhs);
      };

    case HloOpcode::kOr:
      return [](llvm::IRBuilder<>* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateOr(lhs, rhs);
      };

    case HloOpcode::kXor:
      return [](llvm::IRBuilder<>* b, llvm::Value* lhs, llvm::Value* rhs) {
        return b->CreateXor(lhs, rhs);
      };

    case HloOpcode::kMaximum:
      return [root_is_floating_point, root_is_signed](
                 llvm::IRBuilder<>* b, llvm::Value* lhs,
                 llvm::Value* rhs) -> llvm::Value* {
        if (root_is_floating_point) {
          return llvm_ir::EmitCallToIntrinsic(llvm::Intrinsic::maxnum,
                                              {lhs, rhs}, {lhs->getType()}, b);
        }

        return b->CreateSelect(
            b->CreateICmp(root_is_signed ? llvm::ICmpInst::ICMP_SGE
                                         : llvm::ICmpInst::ICMP_UGE,
                          lhs, rhs),
            lhs, rhs);
      };

    case HloOpcode::kMinimum:
      return [root_is_floating_point, root_is_signed](
                 llvm::IRBuilder<>* b, llvm::Value* lhs,
                 llvm::Value* rhs) -> llvm::Value* {
