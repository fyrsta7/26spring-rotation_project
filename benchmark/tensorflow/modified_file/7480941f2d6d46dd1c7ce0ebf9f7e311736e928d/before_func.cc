  LogicalResult matchAndRewrite(mhlo::CstrReshapableOp op,
                                PatternRewriter &rewriter) const override {
    // Get shape analysis info for the number of elements.
    ShapeComponentAnalysis shapeComponentAnalysis;
    auto numElementsInfo =
        shapeComponentAnalysis.GetValueInfo(op.num_elements());
    if (!numElementsInfo) return failure();
    assert(numElementsInfo->size() == 1 && "expect one value for a scalar");
    auto numElements = numElementsInfo->front();

    // Get shape analysis info for the dynamic shape.
    auto dynShapeDims = shapeComponentAnalysis.GetValueInfo(op.dynamic_shape());
    if (!dynShapeDims) return failure();

    // We can handle two cases:
    //   - there is exactly one -1 in the dynamic shape, i.e. a unique wildcard
    //     dimension, or
    //   - there is no -1 in the dynamic shape, i.e. no wildcard dimension.
    bool unique_wildcard_dimension = false;
    for (const auto &d : *dynShapeDims) {
      if (d.isConstant(-1)) {
        if (unique_wildcard_dimension) return failure();
        unique_wildcard_dimension = true;
      } else if (!d.isKnownNotNegativeOne()) {
        return failure();
      }
    }

    // We can only handle simple products with constants and symbols. Find all
    // the factors based on the number of elements.
    int64_t concreteProductNumElems = 1;
    SmallVector<Symbol> remainingSymbolicFactorsNumElems;
    if (!IsSimpleProduct(numElements, &concreteProductNumElems,
                         &remainingSymbolicFactorsNumElems)) {
      return failure();
    }
    assert(concreteProductNumElems >= 1 &&
           "number of elements cannot entail negative or zero factors");

    // Find all factors based on the dynamic shape.
    //   - Accumulate the conrete product to later compare it against its
    //     equivalent based on the number of elements.
    //   - Remove symbolic factors from the list and fail if we find an unknown
    //     factor, i.e. if the symbolic factors based on the dynamic shape are
    //     not a subset of the factors based on the number of elements.
    int64_t concreteProductDynShape = 1;
    for (auto dim : *dynShapeDims) {
      SmallVector<Symbol> partialSymbolicFactorsDynShape;
      if (!IsSimpleProduct(
              dim,
              [&](int64_t c) {
                if (c != -1) concreteProductDynShape *= c;
              },
              [&](Symbol s) { partialSymbolicFactorsDynShape.push_back(s); })) {
        return failure();
      }
      for (const Symbol &symDynShape : partialSymbolicFactorsDynShape) {
        auto *it = llvm::find(remainingSymbolicFactorsNumElems, symDynShape);
        if (it == remainingSymbolicFactorsNumElems.end()) return failure();
        remainingSymbolicFactorsNumElems.erase(it);
      }
    }
    assert(concreteProductDynShape >= 1 &&
           "concrete product must not aggregate negative or zero factors");

    // A wildcard dimension can subsume the remaining symbolic factors and
    // potentially also a concrete factor.
    if (unique_wildcard_dimension) {
      if (concreteProductNumElems % concreteProductDynShape != 0)
        return failure();
      rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, true);
      return success();
    }

    // W/o a wildcard, the symbolic and concrete products must be equal.
    bool isReshapable = remainingSymbolicFactorsNumElems.empty() &&
                        concreteProductNumElems == concreteProductDynShape;
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, isReshapable);
    return success();
  }