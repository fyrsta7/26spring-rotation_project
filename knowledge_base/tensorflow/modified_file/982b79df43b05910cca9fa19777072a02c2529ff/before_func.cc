    Value result = InsertDynamicMemrefCastOp(op, operands.front(), &rewriter);

    if (enforce_identity_maps_(op)) {
      result = CreateCopy(op, result, &rewriter);
    }

    return result;
  }

 private:
  // Inserts dynamic memref to change the layout of the memref to put 0-stride
  // and size of the target dimension if size-1 dimension expansion is
  // necessary.
  memref::ReinterpretCastOp InsertDynamicMemrefCastOp(
      mhlo::DynamicBroadcastInDimOp op, Value operand, OpBuilder* b) const {
    auto loc = op.getLoc();
    auto operand_type = operand.getType().cast<MemRefType>();
    auto operand_shape = operand_type.getShape();
    auto operand_rank = operand_type.getRank();

    auto result_type = op.getType().cast<RankedTensorType>();
    auto result_rank = result_type.getRank();

    Value zero = b->create<arith::ConstantIndexOp>(loc, 0);
    Value one = b->create<arith::ConstantIndexOp>(loc, 1);

    // Compute a reversed scan product. Compute the stride for the dimensions so
    // far, working from minor to major dimensions. Additionally, save the
    // operand shape Values to use in the next loop.
    SmallVector<Value, 2> operand_strides(operand_rank, one);
    SmallVector<Value, 2> operand_sizes(operand_rank, one);
    Value stride_so_far = one;
    for (int i = operand_rank - 1; i >= 0; --i) {
      Value operand_dim_size =
          ShapedType::isDynamic(operand_shape[i])
              ? b->create<memref::DimOp>(loc, operand, i).getResult()
              : b->create<arith::ConstantIndexOp>(loc, operand_shape[i])
                    .getResult();
      operand_sizes[i] = operand_dim_size;

      operand_strides[i] = stride_so_far;
      if (i > 0) {
        stride_so_far =
            b->create<arith::MulIOp>(loc, stride_so_far, operand_dim_size);
      }
    }

    SmallVector<OpFoldResult, 2> sizes, strides;
    sizes.reserve(result_rank);
    strides.reserve(result_rank);

    DenseMap<int, int> output_to_input_dim;
    for (auto dim : llvm::enumerate(op.broadcast_dimensions())) {
      output_to_input_dim[dim.value().getSExtValue()] = dim.index();
    }
    for (int i = 0; i < result_rank; ++i) {
      Value i_val = b->create<arith::ConstantIndexOp>(loc, i);
      Value result_dim_size =
          b->create<tensor::ExtractOp>(loc, op.output_dimensions(), i_val);
      if (!result_dim_size.getType().isIndex()) {
        result_dim_size = b->create<arith::IndexCastOp>(loc, result_dim_size,
                                                        b->getIndexType());
      }
      if (result_type.isDynamicDim(i)) {
        sizes.push_back(result_dim_size);
      } else {
        sizes.push_back(b->getIndexAttr(result_type.getDimSize(i)));
      }

      auto it = output_to_input_dim.find(i);
      // If the rank of the output is greater than the rank of the input, i.e.
      // there was no output dimension in the inverse broadcast_dimensions map
      // we also set stride to 0 to emulate padding of the shape with 1s and the
      // corresponding expansion.
      if (it == output_to_input_dim.end()) {
        strides.push_back(zero);
        continue;
      }

      // There can be two cases:
      // 1) Operand dim == result dim => expansion is not needed
      //    => stride flattened buffer stride
      // 2) Operand dim < result dim => expansion is needed => stride := 0.
      int dim = it->second;
      Value is_expansion = b->create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, operand_sizes[dim], result_dim_size);
      Value select = b->create<mlir::SelectOp>(loc, is_expansion, zero,
                                               operand_strides[dim]);
      strides.push_back(select);
    }

