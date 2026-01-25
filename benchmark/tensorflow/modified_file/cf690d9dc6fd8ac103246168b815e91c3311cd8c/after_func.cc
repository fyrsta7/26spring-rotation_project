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
         * layout using the TPU API. Running legalize_tf.cc on non-TPU nodes
         * thus is a potential source of bugs.
         */
        minor_to_major.resize(t.getRank());
        std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
        std::sort(minor_to_major.begin(), minor_to_major.end(),
                  [=](int64_t a, int64_t b) {
                    int64_t da = t.getDimSize(a);
                    int64_t db = t.getDimSize(b);
                    return da > db || (da == db && a > b);
                  });
      }
      std::vector<Attribute> elements;
      elements.reserve(minor_to_major.size());
      for (auto e : minor_to_major) {
        elements.push_back(rewriter.getIntegerAttr(i64_type, e));
      }
      return rewriter.getArrayAttr(elements);
    } else {
      return rewriter.getUnitAttr();  // e.g. tokens
    }
  }