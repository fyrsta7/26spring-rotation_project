  PCHECK(found != -1) << "waitpid(" << pid_ << ", &status, 0)";
  // Though the child process had quit, this call does not close the pipes
  // since its descendants may still be using them.
  DCHECK_EQ(found, pid_);
  returnCode_ = ProcessReturnCode::make(status);
  pid_ = -1;
  return returnCode_;
}

void Subprocess::waitChecked() {
  wait();
  checkStatus(returnCode_);
}

ProcessReturnCode Subprocess::waitTimeout(TimeoutDuration timeout) {
  returnCode_.enforce(ProcessReturnCode::RUNNING);
  DCHECK_GT(pid_, 0) << "The subprocess has been waited already";

  auto pollUntil = std::chrono::steady_clock::now() + timeout;
  auto sleepDuration = std::chrono::milliseconds{2};
  constexpr auto maximumSleepDuration = std::chrono::milliseconds{100};

  for (;;) {
    // Always call waitpid once after the full timeout has elapsed.
    auto now = std::chrono::steady_clock::now();

    int status;
    pid_t found;
    do {
      found = ::waitpid(pid_, &status, WNOHANG);
    } while (found == -1 && errno == EINTR);
    PCHECK(found != -1) << "waitpid(" << pid_ << ", &status, WNOHANG)";
    if (found) {
      // Just on the safe side, make sure it's the actual pid we are waiting.
      DCHECK_EQ(found, pid_);
      returnCode_ = ProcessReturnCode::make(status);
      // Change pid_ to -1 to detect programming error like calling
