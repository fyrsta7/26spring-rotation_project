    if (current->flatbuffer_tensor->is_variable()) {
      if (current->runtime_tensor->data.uint8 == nullptr) {
        error_reporter_->Report("Variable is not allocated");
        return kTfLiteError;
      }
      tflite::ResetVariableTensor(current->runtime_tensor);
    }
  }

  active_ = false;
  return kTfLiteOk;
}

TfLiteStatus MicroAllocator::InitializeRuntimeTensor(
    const tflite::Tensor& flatbuffer_tensor,
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    ErrorReporter* error_reporter, TfLiteTensor* result) {
  if (!active_) {
    return kTfLiteError;
  }

  // Make sure the serialized type is one we know how to deal with, and convert
  // it from a flatbuffer enum into a constant used by the kernel C API.
  TF_LITE_ENSURE_STATUS(ConvertTensorType(flatbuffer_tensor.type(),
                                          &result->type, error_reporter));
  // Make sure we remember if the serialized tensor is designated as a variable.
  result->is_variable = flatbuffer_tensor.is_variable();

  // We need to figure out where the actual contents of this tensor are stored
  // in memory. We'll check to see if there's a serialized buffer (pretty much
  // the same as a constant op in TensorFlow) associated with this tensor first,
  // and if there is update the runtime structure to point to its location in
  // memory.
  result->data.raw = nullptr;
  result->bytes = 0;
  // First see if there's any buffer information in the serialized tensor.
  if (auto* buffer = (*buffers)[flatbuffer_tensor.buffer()]) {
    // If we've found a buffer, does it have any data?
    if (auto* array = buffer->data()) {
      // If it has any data, is the data size larger than zero?
      if (array->size()) {
        // We've found a buffer with valid data, so update the runtime tensor
        // data structure to point to it.
        result->data.raw =
            const_cast<char*>(reinterpret_cast<const char*>(array->data()));
        // We set the data from a serialized buffer, so record tha.
        result->allocation_type = kTfLiteMmapRo;
      }
    }
    // TODO(petewarden): It's not clear in what circumstances we could have a
    // buffer in the serialized tensor, but it doesn't have any data in it. Is
    // that a validly-generated file, and if so what does it mean, or is it an
    // error condition? It would be good to tighten up the specification to make
    // it less ambiguous.
  }

  // TODO(petewarden): Some of these paths aren't getting enough testing
  // coverage, so we should figure out some tests that exercise them.
  if (!result->data.raw) {
    // The tensor contents haven't been set from a serialized buffer, so
    // make a note that they will be allocated from memory. The actual
    // allocation won't happen until later.
    result->allocation_type = kTfLiteArenaRw;
  }

  // Figure out what the size in bytes of the buffer is and store it.
  size_t type_size;
  TF_LITE_ENSURE_STATUS(BytesRequiredForTensor(
      flatbuffer_tensor, &result->bytes, &type_size, error_reporter));
  // Copy the shape of the tensor from the serialized data into the runtime
  // form. We have to allocate memory for this.
  result->dims =
      reinterpret_cast<TfLiteIntArray*>(memory_allocator_.AllocateFromTail(
          TfLiteIntArrayGetSizeInBytes(flatbuffer_tensor.shape()->Length()),
          alignof(TfLiteIntArray)));
  result->dims->size = flatbuffer_tensor.shape()->Length();
  for (size_t n = 0; n < flatbuffer_tensor.shape()->Length(); ++n) {
    result->dims->data[n] = flatbuffer_tensor.shape()->Get(n);
  }
  // Copy the quantization information from the serialized data.
  const auto* src_quantization = flatbuffer_tensor.quantization();
  if (src_quantization && src_quantization->scale() &&
      (src_quantization->scale()->size() > 0) &&
      src_quantization->zero_point() &&
      (src_quantization->zero_point()->size() > 0)) {
    result->params.scale = src_quantization->scale()->Get(0);
    // This magic handles issues with little-endianness.
    for (unsigned int b = 0; b < sizeof(int64_t); ++b)
      *(reinterpret_cast<char*>(&result->params.zero_point) + b) =
          *(reinterpret_cast<const char*>(
                src_quantization->zero_point()->Data()) +
            b);
    result->params.zero_point =
        flatbuffers::EndianScalar(result->params.zero_point);

    // Populate per-channel quantization params.
    int channels = src_quantization->scale()->size();
    TfLiteAffineQuantization* quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            memory_allocator_.AllocateFromTail(
                sizeof(TfLiteAffineQuantization),
                alignof(TfLiteAffineQuantization)));
    quantization->zero_point =
        reinterpret_cast<TfLiteIntArray*>(memory_allocator_.AllocateFromTail(
            TfLiteIntArrayGetSizeInBytes(channels), alignof(TfLiteIntArray)));
    quantization->scale =
        reinterpret_cast<TfLiteFloatArray*>(memory_allocator_.AllocateFromTail(
            TfLiteFloatArrayGetSizeInBytes(channels),
            alignof(TfLiteFloatArray)));
    quantization->zero_point->size = channels;
    quantization->scale->size = channels;
    int* zero_point_data = quantization->zero_point->data;
    float* scale_data = quantization->scale->data;
    for (int i = 0; i < channels; i++) {
      zero_point_data[i] = src_quantization->zero_point()->Get(i);
      scale_data[i] = src_quantization->scale()->Get(i);
    }
    // TODO(rocky): Need to add a micro_allocator test case that fails when
    // this is not copied:
    quantization->quantized_dimension = src_quantization->quantized_dimension();

    result->quantization = {kTfLiteAffineQuantization, quantization};
  }
