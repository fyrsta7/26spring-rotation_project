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
  for (const auto &it : llvm::enumerate(is_outer_dim)) {
    if (it.value()) {
      outer_dims.push_back(it.index());
    }
  }

  if (outer_dims_first) {
    return TransposeReshape(arg, loc, outer_dims, contract_dims, shape,
                            rewriter);
  }

  return TransposeReshape(arg, loc, contract_dims, outer_dims, shape, rewriter);
}