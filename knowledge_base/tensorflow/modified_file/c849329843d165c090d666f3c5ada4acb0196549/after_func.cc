// %token = "mhlo.create_token"() : () -> !mhlo.token
// %data_and_token = "mhlo.infeed"(%token) {infeed_config = ""} :
//      (!mhlo.token) -> tuple<tuple<tensor<3xi32>, tensor<4xf32>>,
//      !mhlo.token>
// %data = "mhlo.get_tuple_element"(%data_and_token) {index = 0}
// %0#0 = "mhlo.get_tuple_element"(%data) {index = 0}
// %0#1 = "mhlo.get_tuple_element"(%data) {index = 1}
//
class ConvertInfeedDequeueTupleOp
    : public OpRewritePattern<TF::InfeedDequeueTupleOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TF::InfeedDequeueTupleOp op,
                                PatternRewriter &rewriter) const override {
    std::vector<Type> result_types(op.outputs().size());
    for (auto idx_and_output : llvm::enumerate(op.outputs())) {
      result_types[idx_and_output.index()] = (idx_and_output.value().getType());
    }

    for (Type t : result_types) {
      if (auto ty = t.dyn_cast<RankedTensorType>()) {
        if (!ty.hasStaticShape()) return failure();
      }
    }

    // Infeed takes a single token operand. Generate the token using
    // create_token op to pass to the infeed op.
    auto token = rewriter.create<CreateTokenOp>(
        op.getLoc(), mhlo::TokenType::get(rewriter.getContext()));

    // Emit infeed op.
    // The result type of infeed is a tuple(tuple(result types), token type).
    auto data_tuple_type =
        mlir::TupleType::get(rewriter.getContext(), result_types);
    auto data_and_token_type = mlir::TupleType::get(
        rewriter.getContext(), {data_tuple_type, token.getType()});

    ArrayAttr layout;  // filled in during the xla-adjust-layout pass

    auto data_and_token =
        rewriter.create<InfeedOp>(op.getLoc(), data_and_token_type, token,
                                  /*infeed_config=*/rewriter.getStringAttr(""),
                                  /*layout=*/layout);

    if (op._XlaSharding().hasValue()) {
      // _XlaSharding attribute in TF is a serialized string of the OpSharding
      // proto, so convert to a text form here.
      ::xla::OpSharding sharding_proto;
      if (!sharding_proto.ParseFromString(op._XlaSharding().getValue().str()))
        return failure();

      // Token is a control signal and not a real data, so arbitrarily assign
      // the token to device 0.
      if (sharding_proto.type() == ::xla::OpSharding::TUPLE) {
        *sharding_proto.add_tuple_shardings() =
            ::xla::sharding_builder::AssignDevice(0);
        data_and_token->setAttr(
            kShardingAttr,
            rewriter.getStringAttr(sharding_proto.SerializeAsString()));
      } else {
        data_and_token->setAttr(kShardingAttr, op._XlaShardingAttr());
      }
    }

    // The infeed instruction produces a tuple of the infeed data and a token
    // type. Emit get_tuple_element to get infeed data tuple.
    auto data_tuple = rewriter.create<GetTupleElementOp>(
        op.getLoc(), data_tuple_type, data_and_token,
