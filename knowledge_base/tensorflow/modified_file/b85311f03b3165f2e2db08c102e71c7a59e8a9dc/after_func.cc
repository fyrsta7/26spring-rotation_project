
Value getConstantLike(OpBuilder& b, Location loc, const APFloat& constant,
                      Value val) {
  Type ty = getElementTypeOrSelf(val.getType());
  return b.create<ConstantLikeOp>(loc, b.getFloatAttr(ty, constant), val);
}

//===----------------------------------------------------------------------===//
// BinaryOps
//===----------------------------------------------------------------------===//

namespace {
// Gets the resulting type from a broadcast between two types.
static Type GetBroadcastType(Type x, Type y, Type element_type,
                             DenseIntElementsAttr broadcast_dimensions_attr) {
  auto x_ranked = x.dyn_cast<RankedTensorType>();
  auto y_ranked = y.dyn_cast<RankedTensorType>();
  if (!x_ranked || !y_ranked) {
    return UnrankedTensorType::get(element_type);
  }

  auto shape_x = x_ranked.getShape();
  auto shape_y = y_ranked.getShape();

  if (shape_x.size() == shape_y.size()) {
    llvm::SmallVector<int64_t, 4> out_shape(shape_x.size());
    for (int i = 0, e = shape_x.size(); i < e; i++) {
      auto x_val = shape_x[i];
      auto y_val = shape_y[i];
      if (x_val == -1 || y_val == -1) {
        out_shape[i] = -1;
      } else {
        out_shape[i] = std::max(x_val, y_val);
      }
    }
    return RankedTensorType::get(out_shape, element_type);
  }

  auto shape_large = shape_x.size() > shape_y.size() ? shape_x : shape_y;
  auto shape_small = shape_x.size() <= shape_y.size() ? shape_x : shape_y;

  llvm::SmallVector<int64_t, 4> broadcast_dimensions;
  if (broadcast_dimensions_attr) {
    // Explicit broadcast dimensions.
    for (const APInt& int_value :
         broadcast_dimensions_attr.getValues<APInt>()) {
      broadcast_dimensions.push_back(int_value.getSExtValue());
    }
    if (broadcast_dimensions.size() != shape_small.size()) {
      // Signal illegal broadcast_dimensions as unranked.
      return UnrankedTensorType::get(element_type);
    }
  } else {
    // If no broadcast dimensions, assume "numpy" broadcasting.
    broadcast_dimensions = llvm::to_vector<4>(llvm::seq<int64_t>(
        shape_large.size() - shape_small.size(), shape_large.size()));
  }

  llvm::SmallVector<int64_t, 4> out_shape(shape_large.begin(),
