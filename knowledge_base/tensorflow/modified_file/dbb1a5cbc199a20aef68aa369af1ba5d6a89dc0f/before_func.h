  using OutputConstMatrixMap = typename Base::OutputConstMatrixMap;
  using OutputConstMatrixMaps = typename Base::OutputConstMatrixMaps;

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    int64_t n = input_matrix_shapes[0].dim_size(0);
    if (compute_v_) {
      return TensorShapes({TensorShape({n}), TensorShape({n, n})});
    } else {
      return TensorShapes({TensorShape({n})});
    }
  }

  void ComputeMatrix(OpKernelContext* context,
                     const InputConstMatrixMaps& inputs,
                     OutputMatrixMaps* outputs) final {
    const int64_t rows = inputs[0].rows();
    if (rows == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      return;
    }

    // This algorithm relies on denormals, so switch them back on locally.
    port::ScopedDontFlushDenormal dont_flush_denormals;

    Eigen::ComplexEigenSolver<OutputMatrix> eig(
