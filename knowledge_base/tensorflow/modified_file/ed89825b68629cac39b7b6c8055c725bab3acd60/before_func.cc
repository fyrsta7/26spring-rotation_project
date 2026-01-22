// kernels contributed by ARM here,
//
//     https://github.com/google/gemmlowp/pull/116
//
// However, as of April 2018, there don't seem to be any commercially available
// CPU supporting these instructions (yet); we are waiting for
// Cortex-A{75,55}-r1 to become available; the "-r1" is key here. Even if such
// CPUs become available soon, it will presumably take years for them to
// overtake the large volume of existing CPUs not supporting these new
// instructions, especially in current and future low-end devices. All in all,
// we can foresee these 'fast int8 kernels' to remain important to have into
// the 2020s.
//
::tensorflow::Status EnsureUint8WeightsSafeForFastInt8Kernels::Run(
    Model* model, std::size_t op_index, bool* modified) {
  *modified = false;
  const auto& op = *model->operators[op_index];
  int weights_index = 0;
  switch (op.type) {
    case OperatorType::kConv:
      weights_index = 1;
      break;
    case OperatorType::kLstmCell:
      weights_index = 2;
      break;
    case OperatorType::kFullyConnected: {
      weights_index = 1;
      const auto& fc_op = static_cast<const toco::FullyConnectedOperator&>(op);
      CHECK(fc_op.weights_format == FullyConnectedWeightsFormat::kDefault)
          << "This graph transformation expects to run before FC weights get "
             "shuffled.";
      break;
    }
    default:
      // Other operator types are unaffected by this graph transformation,
      // because their runtime implementations don't use the fast int8 trick.
      // In particular that's the case of DepthwiseConv at the moment.
      // We have to update this logic when that changes, e.g. if in the future
      // some DepthwiseConv kernel wants to use the trick.
      //
      // The reason why that's not so likely, hence why it's fairly safe to
      // stay conservative in the list of operators that we handle here, is that
      // the fast int8 kernel trick is only applicable to ops that either are
      // implemented as a GEMM, or use symmetric ranges for both weights and
      // activations. The reason why GEMM is special (can use the trick even
      // without symmetric ranges) is that it is so arithmetic-intense that
      // it can use techniques reducing its implementation to the symmetric
      // ranges case, with limited relative overhead (O(N^2) overhead vs
      // O(N^3) GEMM cost). See https://arxiv.org/pdf/1712.05877, section
      // 2.3 Efficient handling of zero-points.
      //
      // That's why at the moment we only handle operators that use a GEMM
      // (Conv, fully-connected --- note that LSTM merely wraps a
      // fully-connected operator).
      return ::tensorflow::Status::OK();
  }

  const string& name = op.inputs[weights_index];
  auto& array = model->GetArray(name);
  if (!array.buffer) {
    return ::tensorflow::Status::OK();
  }
  if (array.data_type != ArrayDataType::kUint8) {
    return ::tensorflow::Status::OK();
  }
  auto& buffer_data = array.GetMutableBuffer<ArrayDataType::kUint8>().data;

  int count_bad = 0;
  int index_of_previous_bad_value = 0;
  bool changed = false;

  for (size_t i = 0; i < buffer_data.size(); i++) {
    if (buffer_data[i] == 0) {
      count_bad++;
      if (count_bad > 1) {
        const int distance = i - index_of_previous_bad_value;
        // Semi-arbitrary threshold. The idea is that trouble only occurs
        // when two bad values are very close to each other so that they
        // are jointly used within registers inside some GEMM kernel.
        // The details of that depend on the kernel. Our current fast ARM64
        // kernel, for instance, only has an issue when the distance between
        // consecutive bad values is exactly 8. We do not want to track such
        // kernel details too closely here, so we pick a threshold that's
        // a bit larger than that, to give us room to change kernels in the
        // future without worrying.
        static constexpr int kMinDistanceBetweenBadValues = 16;
        if (distance < kMinDistanceBetweenBadValues) {
          if (allow_nudging_weights() || has_default_ranges_flag()) {
            buffer_data[i] = 1;
            changed = true;
            continue;
          }
          LOG(FATAL) << "Bad value for " << name << " at index " << i
                     << ", previous bad value at index "
                     << index_of_previous_bad_value << ", distance=" << distance
                     << ", kMinDistanceBetweenBadValues="
                     << kMinDistanceBetweenBadValues << ". Consider passing "
                     << "--allow_nudging_weights_to_use_fast_gemm_kernel "
                     << "if you don't care about accuracy.";
        }
      }
      index_of_previous_bad_value = i;
    }
  }

  if (changed) {
    if (has_default_ranges_flag()) {
      std::cerr
