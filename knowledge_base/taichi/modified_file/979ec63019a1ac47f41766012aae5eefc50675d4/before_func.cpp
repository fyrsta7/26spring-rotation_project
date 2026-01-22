uint64 Program::fetch_result_uint64(int i) {
  uint64 ret;
  auto arch = config.arch;
  if (arch == Arch::cuda) {
    // TODO: refactor
    // We use a `memcpy_device_to_host` call here even if we have unified
    // memory. This simplifies code. Also note that a unified memory (4KB) page
    // fault is rather expensive for reading 4-8 bytes.
#if defined(TI_WITH_CUDA)
    CUDADriver::get_instance().memcpy_device_to_host(
        &ret, (uint64 *)result_buffer + i, sizeof(uint64));
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
