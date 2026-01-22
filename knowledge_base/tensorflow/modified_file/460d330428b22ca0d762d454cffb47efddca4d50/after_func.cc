  for (auto val : transpose_permutation) {
    transposed_shape.push_back(arg_shape[val]);
  }
  auto transpose_type = RankedTensorType::get(transposed_shape, element_type);
  auto transpose_result = rewriter->create<TransposeOp>(
      loc, transpose_type, arg, transpose_permutation_attr);

  // Return the final result.
  auto reshaped_type =
      RankedTensorType::get({left_size, right_size}, element_type);
  return rewriter->create<ReshapeOp>(loc, reshaped_type, transpose_result);
}

Value ProcessDotArg(Value arg, Location loc,
                    ArrayRef<int64_t> contract_dims_attr, bool outer_dims_first,
                    PatternRewriter *rewriter) {
  auto shape = arg.getType().cast<ShapedType>().getShape();

  llvm::SmallVector<bool, 5> is_outer_dim;
  is_outer_dim.resize(shape.size(), true);

  // Compute the contract dimension ordering.
  llvm::SmallVector<int64_t, 5> contract_dims;
  for (auto dim : contract_dims_attr) {
    contract_dims.push_back(dim);
    is_outer_dim[dim] = false;
  }

  // Compute the outer dimension orderings.
  llvm::SmallVector<int64_t, 5> outer_dims;
