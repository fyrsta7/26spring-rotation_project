  : file_(std::move(file)),
    fileId_(fileId),
    writeLock_(file_, std::defer_lock),
    filePos_(0) {
  if (!writeLock_.try_lock()) {
    throw std::runtime_error("RecordIOWriter: file locked by another process");
  }

  struct stat st;
  checkUnixError(fstat(file_.fd(), &st), "fstat() failed");

  filePos_ = st.st_size;
}

void RecordIOWriter::write(std::unique_ptr<IOBuf> buf) {
