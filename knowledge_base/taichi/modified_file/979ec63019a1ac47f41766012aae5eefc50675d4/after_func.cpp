uint64 Program::fetch_result_uint64(int i) {
  uint64 ret;
  auto arch = config.arch;
  synchronize();
  if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    if (config.use_unified_memory) {
      // More efficient than a cudaMemcpy call in practice
      ret = ((uint64 *)result_buffer)[i];
    } else {
      CUDADriver::get_instance().memcpy_device_to_host(
          &ret, (uint64 *)result_buffer + i, sizeof(uint64));
    }
#else
    TI_NOT_IMPLEMENTED;
#endif
  } else if (arch_is_cpu(arch)) {
    ret = ((uint64 *)result_buffer)[i];
  } else {
    ret = context.get_arg_as_uint64(i);
  }
  return ret;
}
