  using ResultType = tfrt_stub::FallbackTensor;
  using ConversionContext = TensorflowConversionContext;

  template <typename T, int rank>
  static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, rank>* memref) {
    return memref->sizes;
  }

  template <typename T>
  static llvm::ArrayRef<int64_t> Sizes(StridedMemRefType<T, 0>* memref) {
    return {};
  }

  template <typename T, int rank>
  static Tensor Convert(ConversionContext& ctx, void* memref_ptr) {
    auto* memref = static_cast<StridedMemRefType<T, rank>*>(memref_ptr);
    auto memref_sizes = Sizes(memref);

    // Convert TFRT data type into Tensorflow data type.
    auto dtype = tfd::GetTfDataType(tfrt::GetDType<T>());

    // Build a Tensorflow TensorShape from memref sizes.
    TensorShape shape(memref_sizes);

    // Check if returned memref already has corresponding runtime tensor.
    auto it = ctx.runtime_tensors.find(memref->data);
    ConversionContext::TensorOrBuffer runtime_tensor =
        it != ctx.runtime_tensors.end() ? it->second : nullptr;

    // Forward operand tensor to the result.
    if (auto* operand = runtime_tensor.dyn_cast<const Tensor*>()) {
      Tensor result;
      auto st = result.BitcastFrom(*operand, dtype, shape);
      assert(st.ok() && "failed to bitcast from forwarded tensor");
      (void)st;
      return result;
    }

    // The same memref returned multiple times.
    if (auto* buffer = runtime_tensor.dyn_cast<TensorBuffer*>()) {
      buffer->Ref();
      auto ptr = core::RefCountPtr<TensorBuffer>(buffer);
      return Tensor(dtype, std::move(shape), std::move(ptr));
    }

    // This is a newly allocated memref, and we need to wrap it into the runtime
    // tensor buffer to pass it back to the caller as a Tensor.
    size_t size = sizeof(T);
    for (int i = 0; i < rank; ++i) size *= memref_sizes[i];

    // Create a TensorBuffer from the returned memref.
    TF_ANNOTATE_MEMORY_IS_INITIALIZED(memref->data, size);
    auto* buffer = new MemrefTensorBuffer(
        memref->basePtr, memref->data, size,
        /*owner=*/!internal::IsStaticStorageDuration(memref));

    // Construct a tensor from the memory buffer.
    auto ptr = core::RefCountPtr<MemrefTensorBuffer>(buffer);
    Tensor tensor(dtype, std::move(shape), std::move(ptr));

    // Keep track of memrefs already used to construct runtime tensors.
