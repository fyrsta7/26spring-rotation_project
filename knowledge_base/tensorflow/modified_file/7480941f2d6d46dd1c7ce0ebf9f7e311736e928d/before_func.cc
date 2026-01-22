    if (dim.isConstant(1)) continue;
    if (inputDim >= operand_shape->size() || dim != (*operand_shape)[inputDim])
      return false;
    ++inputDim;
  }
  return inputDim == operand_shape->size();
}

// Rewrite dynamic reshapes that only insert one dimensions into
// tensor.expand_shape.
struct ReshapeToExpandShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  ReshapeToExpandShape(MLIRContext *ctx) : OpRewritePattern(ctx) {}
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    ShapeComponentAnalysis shapeComponentAnalysis;
    if (!isExpandShape(shapeComponentAnalysis, op)) return failure();
    auto output_shape = shapeComponentAnalysis.GetValueInfo(op.output_shape());
    SmallVector<ReassociationExprs> reassociations(output_shape->size());
    auto *it = reassociations.begin();
