    if (!tensorflow::tpu::IsInitialized(executor)) {
      return failure();
    }
    ApiConverter::ToC(old_shape, &old_shape_c);
    executor->TpuTransferManager_GetInfeedLayoutFn(&old_shape_c, &new_shape_c);
    xla::Shape new_shape = ApiConverter::FromC(&new_shape_c);
    ApiConverter::Free(&old_shape_c);
    ApiConverter::Free(&new_shape_c);

    auto minor_to_major = new_shape.layout().minor_to_major();
    return std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end());
  }

  static FailureOr<Attribute> GetLayout(const Type &type, OpBuilder &rewriter) {
    auto i64_type = rewriter.getIntegerType(64);
    if (type.isa<TupleType>()) {
      auto tuple_type = type.dyn_cast<TupleType>();
      auto types = tuple_type.getTypes();
      llvm::SmallVector<mlir::Attribute, types.size()> v;
      for (const mlir::Type &t : types) {
        auto layout = GetLayout(t, rewriter);
        if (failed(layout)) return failure();
        v.push_back(layout.getValue());
      }
      ArrayRef<Attribute> shape(v);
      return rewriter.getArrayAttr(shape);
    } else if (auto t = type.dyn_cast<RankedTensorType>()) {
      if (!t.hasStaticShape()) return failure();
      auto layout = GetTPUInfeedLayoutFromAPI(t);
      std::vector<int64_t> minor_to_major;
      if (succeeded(layout)) {
        minor_to_major = layout.getValue();
      } else {
        /* If we're not running on a TPU node, we might not be able to
         * actually call the part of the TPU API that gives us layout.
         * This happens e.g. for unit tests. Below we just create a reasonable
         * layout.  We sort by dimension size, which makes the layout agree with
         * the "correct" TPU layout in surprisingly many cases.
         * Note that the corresponding InfeedEnqueue op will be generated
         * through another path, and might still generate an (incompatible)
