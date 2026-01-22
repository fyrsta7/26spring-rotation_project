  return absl::c_count_if(
      fusion.getFusionParameters(), [&](mlir::Value parameter) {
        Shape parameter_shape = TypeToShape(parameter.getType());
        return ShapeUtil::ElementsIn(parameter_shape) > num_elements;
      });
}

// The benefit of unrolling a kInput fusion that is a column reduction comes
// from the vectorization of non-reduction fusion outputs and fusion inputs.
// On the other hand, unrolling can also introduce factors that can cause
// the kernel to run slower. This routine uses a simple heuristic to estimate
// the benefit as well as the overhead of unrolling in order to decide whether
// unrolling is beneficial for the given kInput fusion.
bool IsUnrollingColumnReductionBeneficial(
    mlir::Operation* unnested_hlo, const Shape& input_shape,
    int64 num_kept_minor, const FusionLayoutAnalysis& layout_analysis) {
  // TODO(b/122468062): Need further investigate to see whether we can
  // remove the constraint on IsPowerOfTwo.
  if (!IsPowerOfTwo(static_cast<uint64>(num_kept_minor))) {
    return false;
  }

  if (IsReductionFromOrToContiguousDimensions(unnested_hlo, layout_analysis)) {
    return true;
  }

  auto fusion = mlir::cast<mlir::lmhlo::FusionOp>(unnested_hlo);
  int64 can_be_vectorized = 0;
  int64 cannot_be_vectorized = 0;
  auto fusion_results = ToStdVector(fusion.getFusionResults());
  absl::flat_hash_set<mlir::Operation*> use_chain_endings;
  if (fusion_results.size() == 1) {
    if (IsReductionFromOrToContiguousDimensions(
            fusion_results[0].getDefiningOp(), layout_analysis)) {
      use_chain_endings.insert(fusion_results[0].getDefiningOp());
      // Atomic.add of the reduction result can't be vectorized.
      cannot_be_vectorized++;
    }
  } else {
    for (mlir::Value result : fusion_results) {
      if (IsReductionFromOrToContiguousDimensions(result.getDefiningOp(),
                                                  layout_analysis)) {
        // Atomic.add of the reduction result can't be vectorized.
        cannot_be_vectorized++;
      } else {
        // Write of the non-reduction result can be vectorized.
        can_be_vectorized++;
      }
      use_chain_endings.insert(result.getDefiningOp());
    }
