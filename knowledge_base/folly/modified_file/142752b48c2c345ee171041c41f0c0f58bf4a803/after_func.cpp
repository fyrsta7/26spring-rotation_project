  , writeTimeout_(this, evb)
  , ioHandler_(this, evb) {
  VLOG(5) << "new AsyncSocket(" << this << ", evb=" << evb << ")";
  init();
  connect(nullptr, ip, port, connectTimeout);
}

AsyncSocket::AsyncSocket(EventBase* evb, int fd)
  : eventBase_(evb)
  , writeTimeout_(this, evb)
  , ioHandler_(this, evb, fd) {
  VLOG(5) << "new AsyncSocket(" << this << ", evb=" << evb << ", fd="
          << fd << ")";
  init();
  fd_ = fd;
  state_ = StateEnum::ESTABLISHED;
