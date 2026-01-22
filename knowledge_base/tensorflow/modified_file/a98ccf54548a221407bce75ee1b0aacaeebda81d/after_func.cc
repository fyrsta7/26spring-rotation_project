
  absl::MutexLock lock(&mutex_);
  auto it = kernels_cache_.find(key);
  if (it != kernels_cache_.end()) return it->second.get();

  auto emplaced = kernels_cache_.try_emplace(key, std::move(kernel));
  return emplaced.first->second.get();
}

//===----------------------------------------------------------------------===//
// Define the kernel launch custom call.
//===----------------------------------------------------------------------===//

static absl::Status LaunchFunc(
    const ServiceExecutableRunOptions* run_options, const std::string* ptx,
    const std::vector<uint8_t>* cubin, se::DeviceMemoryBase* temp_buffer,
    GpuExecutableKernelsCache* kernels_cache, int32_t grid_size_x,
    int32_t grid_size_y, int32_t grid_size_z, int32_t block_size_x,
    int32_t block_size_y, int32_t block_size_z, CustomCall::RemainingArgs args,
    std::string_view name) {
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* executor = stream->parent();

  LaunchDimensions launch_dimensions(
      {grid_size_x, grid_size_y, grid_size_z},
      {block_size_x, block_size_y, block_size_z});

  se::KernelBase* kernel = kernels_cache->Get(executor, name);
  const int args_size_including_temp_buffer = args.size() + 1;

  // If kernel does not exists create it from the ptx and dubin.
  if (kernel == nullptr) {
    auto created =
        CreateKernel(absl::string_view(name.data(), name.size()),
                     args_size_including_temp_buffer, *ptx, *cubin, executor);
    if (!created.ok()) return ToAbslStatus(created.status());

    kernel = kernels_cache->Set(executor, name, std::move(*created));
  }

  VLOG(3) << "Launching " << kernel->name();
  absl::InlinedVector<se::DeviceMemoryBase, 8> buffer_args(
      args_size_including_temp_buffer);

  // Add MemRef arguments as buffer arguments.
  for (unsigned i = 0; i < args.size(); ++i) {
    // Simple row major memref passed as shapeless buffer.
    if (auto memref = args.get<FlatMemrefView>(i); succeeded(memref)) {
      buffer_args[i] = GetDeviceAddress(*memref);
      continue;
    }

    // Memref layout must be encoded in the compiled device kernel, so we don't
    // have to pass strides or minor to major dimensions order to the kernel.
    if (auto strided = args.get<StridedMemrefView>(i); succeeded(strided)) {
      buffer_args[i] = GetDeviceAddress(*strided);
      continue;
    }

    return absl::InvalidArgumentError(
