    rewriter.replaceOp(op, linalg_op.getOperation()->getResults());
    return success();
  }
};

/// Pattern to convert BroadcastOp to Linalg ops.
template <typename OpTy>
class BroadcastConverter
    : public DataMovementOpConverter<BroadcastConverter<OpTy>, OpTy> {
 public:
  using DataMovementOpConverter<BroadcastConverter,
                                OpTy>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy broadcast_op,
                                                   Builder* b) {
    ShapedType input_type =
        broadcast_op.operand().getType().template cast<ShapedType>();
    unsigned input_rank = input_type.getRank();
    unsigned nloops = GetHloOpResultType(broadcast_op).getRank();

    // BroadcastOp prepends the dimensions in the `broadcast_sizes` attribute to
    // the input's dimensions.
    unsigned num_prepended_dims = llvm::size(broadcast_op.broadcast_sizes());
    SmallVector<AffineExpr, 4> input_dim_exprs;
    input_dim_exprs.reserve(input_rank);
    for (unsigned i = 0; i < input_rank; ++i) {
      input_dim_exprs.push_back(b->getAffineDimExpr(num_prepended_dims + i));
