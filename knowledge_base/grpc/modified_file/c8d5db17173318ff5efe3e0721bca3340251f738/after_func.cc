static void tcp_destroy(grpc_endpoint* ep) {
  grpc_network_status_unregister_endpoint(ep);
  grpc_tcp* tcp = reinterpret_cast<grpc_tcp*>(ep);
  grpc_slice_buffer_reset_and_unref_internal(&tcp->last_read_buffer);
  if (grpc_event_engine_can_track_errors()) {
    gpr_atm_no_barrier_store(&tcp->stop_error_notification, true);
    grpc_fd_set_error(tcp->em_fd);
  }
  TCP_UNREF(tcp, "destroy");
}

static void call_read_cb(grpc_tcp* tcp, grpc_error* error) {
  grpc_closure* cb = tcp->read_cb;

  if (grpc_tcp_trace.enabled()) {
    gpr_log(GPR_INFO, "TCP:%p call_cb %p %p:%p", tcp, cb, cb->cb, cb->cb_arg);
    size_t i;
    const char* str = grpc_error_string(error);
    gpr_log(GPR_INFO, "read: error=%s", str);

    for (i = 0; i < tcp->incoming_buffer->count; i++) {
      char* dump = grpc_dump_slice(tcp->incoming_buffer->slices[i],
                                   GPR_DUMP_HEX | GPR_DUMP_ASCII);
      gpr_log(GPR_INFO, "READ %p (peer=%s): %s", tcp, tcp->peer_string, dump);
      gpr_free(dump);
    }
  }

  tcp->read_cb = nullptr;
  tcp->incoming_buffer = nullptr;
  GRPC_CLOSURE_SCHED(cb, error);
}

#define MAX_READ_IOVEC 4
static void tcp_do_read(grpc_tcp* tcp) {
  GPR_TIMER_SCOPE("tcp_do_read", 0);
  struct msghdr msg;
  struct iovec iov[MAX_READ_IOVEC];
  ssize_t read_bytes;
  size_t i;

  GPR_ASSERT(tcp->incoming_buffer->count <= MAX_READ_IOVEC);

  for (i = 0; i < tcp->incoming_buffer->count; i++) {
    iov[i].iov_base = GRPC_SLICE_START_PTR(tcp->incoming_buffer->slices[i]);
    iov[i].iov_len = GRPC_SLICE_LENGTH(tcp->incoming_buffer->slices[i]);
  }

  msg.msg_name = nullptr;
  msg.msg_namelen = 0;
  msg.msg_iov = iov;
  msg.msg_iovlen = static_cast<msg_iovlen_type>(tcp->incoming_buffer->count);
  msg.msg_control = nullptr;
  msg.msg_controllen = 0;
  msg.msg_flags = 0;

  GRPC_STATS_INC_TCP_READ_OFFER(tcp->incoming_buffer->length);
  GRPC_STATS_INC_TCP_READ_OFFER_IOV_SIZE(tcp->incoming_buffer->count);

  do {
    GPR_TIMER_SCOPE("recvmsg", 0);
    GRPC_STATS_INC_SYSCALL_READ();
    read_bytes = recvmsg(tcp->fd, &msg, 0);
  } while (read_bytes < 0 && errno == EINTR);

  if (read_bytes < 0) {
    /* NB: After calling call_read_cb a parallel call of the read handler may
