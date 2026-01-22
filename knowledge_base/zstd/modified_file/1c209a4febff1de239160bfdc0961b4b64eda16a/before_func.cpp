  auto size = file_size(file, ec);
  if (ec) {
    size = 0;
  }
  return size;
}

static size_t handleOneInput(const Options &options,
                             const std::string &inputFile,
                             FILE* inputFd,
                             const std::string &outputFile,
                             FILE* outputFd,
                             ErrorHolder &errorHolder) {
  auto inputSize = fileSizeOrZero(inputFile);
  // WorkQueue outlives ThreadPool so in the case of error we are certain
  // we don't accidently try to call push() on it after it is destroyed.
  WorkQueue<std::shared_ptr<BufferWorkQueue>> outs{2 * options.numThreads};
  size_t bytesWritten;
  {
    // Initialize the thread pool with numThreads + 1
    // We add one because the read thread spends most of its time waiting.
    // This also sets the minimum number of threads to 2, so the algorithm
    // doesn't deadlock.
    ThreadPool executor(options.numThreads + 1);
    if (!options.decompress) {
      // Add a job that reads the input and starts all the compression jobs
      executor.add(
          [&errorHolder, &outs, &executor, inputFd, inputSize, &options] {
            asyncCompressChunks(
                errorHolder,
                outs,
                executor,
                inputFd,
                inputSize,
                options.numThreads,
                options.determineParameters());
          });
      // Start writing
      bytesWritten = writeFile(errorHolder, outs, outputFd, options.decompress);
    } else {
      // Add a job that reads the input and starts all the decompression jobs
      executor.add([&errorHolder, &outs, &executor, inputFd] {
        asyncDecompressFrames(errorHolder, outs, executor, inputFd);
