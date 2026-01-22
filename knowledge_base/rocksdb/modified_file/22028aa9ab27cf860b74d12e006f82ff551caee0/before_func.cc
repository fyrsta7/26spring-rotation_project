bool FilePrefetchBuffer::TryReadFromCache(uint64_t offset, size_t n,
                                          Slice* result, bool for_compaction) {
  if (track_min_offset_ && offset < min_offset_read_) {
    min_offset_read_ = static_cast<size_t>(offset);
  }
  if (!enable_ || offset < buffer_offset_) {
    return false;
  }

  // If the buffer contains only a few of the requested bytes:
  //    If readahead is enabled: prefetch the remaining bytes + readadhead bytes
  //        and satisfy the request.
  //    If readahead is not enabled: return false.
  if (offset + n > buffer_offset_ + buffer_.CurrentSize()) {
    if (readahead_size_ > 0) {
      assert(file_reader_ != nullptr);
      assert(max_readahead_size_ >= readahead_size_);

      Status s =
          Prefetch(file_reader_, offset, n + readahead_size_, for_compaction);
      if (!s.ok()) {
        return false;
      }
      readahead_size_ = std::min(max_readahead_size_, readahead_size_ * 2);
    } else {
      return false;
    }
  }

  uint64_t offset_in_buffer = offset - buffer_offset_;
  *result = Slice(buffer_.BufferStart() + offset_in_buffer, n);
  return true;
}
