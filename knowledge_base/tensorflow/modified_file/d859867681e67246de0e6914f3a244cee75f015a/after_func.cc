              stream,
              /*prefer_to_retain_reference=*/true);
}

StatusOr<ShapedBuffer> PjRtStreamExecutorBuffer::AsShapedBuffer() const {
  absl::MutexLock lock(&mu_);
  if (device_buffer_ == nullptr) {
    return InvalidArgument(
        "Attempted to fetch value of invalid/deleted buffer.");
  }
  return device_buffer_->AsShapedBuffer(on_device_shape_);
}

PjRtStreamExecutorBuffer::ScopedHold
