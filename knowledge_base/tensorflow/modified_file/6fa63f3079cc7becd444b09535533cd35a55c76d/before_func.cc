
  // Broadcast `lhs` and `rhs` to `output_shape`.
  TF_ASSIGN_OR_RETURN(XlaOp lhs_result,
                      DegenerateBroadcastWithUnbounded(
                          builder, lhs, output_dimensions, output_shape));
  TF_ASSIGN_OR_RETURN(XlaOp rhs_result,
                      DegenerateBroadcastWithUnbounded(
                          builder, rhs, output_dimensions, output_shape));
  return UnboundedBroadcastResult{lhs_result, rhs_result};
}

}  // namespace

XlaOp XlaBuilder::BinaryOp(HloOpcode binop, XlaOp lhs, XlaOp rhs,
                           absl::Span<const int64_t> broadcast_dimensions,
                           std::optional<ComparisonDirection> direction,
                           std::optional<Comparison::Type> type) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape* lhs_shape, GetShapePtr(lhs));
    TF_ASSIGN_OR_RETURN(const Shape* rhs_shape, GetShapePtr(rhs));
    TF_ASSIGN_OR_RETURN(
        Shape shape, ShapeInference::InferBinaryOpShape(
                         binop, *lhs_shape, *rhs_shape, broadcast_dimensions));

    XlaOp updated_lhs = lhs;
    XlaOp updated_rhs = rhs;
    if (!lhs_shape->is_unbounded_dynamic() &&
        !rhs_shape->is_unbounded_dynamic()) {
      if (lhs_shape->rank() < shape.rank()) {
        TF_ASSIGN_OR_RETURN(updated_lhs,
                            BroadcastToTargetRank(lhs, *lhs_shape, shape,
                                                  broadcast_dimensions));
      }
      if (rhs_shape->rank() < shape.rank()) {
        TF_ASSIGN_OR_RETURN(updated_rhs,
                            BroadcastToTargetRank(rhs, *rhs_shape, shape,
                                                  broadcast_dimensions));
      }
      TF_ASSIGN_OR_RETURN(const Shape* updated_lhs_shape,
                          GetShapePtr(updated_lhs));
      TF_ASSIGN_OR_RETURN(const Shape* updated_rhs_shape,
                          GetShapePtr(updated_rhs));
      if (!ShapeUtil::SameDimensions(shape, *updated_lhs_shape)) {
        TF_ASSIGN_OR_RETURN(updated_lhs,
                            AddBroadcastSequence(shape, updated_lhs));
      }
      if (!ShapeUtil::SameDimensions(shape, *updated_rhs_shape)) {
        TF_ASSIGN_OR_RETURN(updated_rhs,
                            AddBroadcastSequence(shape, updated_rhs));
      }
    } else {
      if (ShapeUtil::IsScalar(*lhs_shape) || ShapeUtil::IsScalar(*rhs_shape)) {
        if (ShapeUtil::IsScalar(*lhs_shape)) {
          TF_ASSIGN_OR_RETURN(updated_lhs,
                              BroadcastScalarToOutputShapeWithUnbounded(
                                  this, lhs, rhs, *rhs_shape));
        }
        if (ShapeUtil::IsScalar(*rhs_shape)) {
          TF_ASSIGN_OR_RETURN(updated_rhs,
                              BroadcastScalarToOutputShapeWithUnbounded(
                                  this, rhs, lhs, *lhs_shape));
        }
      } else {
        Shape output_shape = shape;
        output_shape.set_element_type(lhs_shape->element_type());
        TF_ASSIGN_OR_RETURN(UnboundedBroadcastResult broadcast_result,
                            BroadcastToOutputShapeWithUnbounded(
                                this, lhs, *lhs_shape, rhs, *rhs_shape,
                                output_shape, broadcast_dimensions));
        updated_lhs = broadcast_result.lhs;
        updated_rhs = broadcast_result.rhs;
      }
    }

    if (binop == HloOpcode::kCompare) {
      if (!direction.has_value()) {
        return InvalidArgument(
            "kCompare expects a ComparisonDirection, but none provided.");
      }
      if (type == std::nullopt) {
        return Compare(shape, updated_lhs, updated_rhs, *direction);
