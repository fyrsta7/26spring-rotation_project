Socket::Socket(
    seastar::connected_socket &&_socket,
    side_t _side,
    uint16_t e_port,
    construct_tag)
  : sid{seastar::this_shard_id()},
    socket(std::move(_socket)),
    in(socket.input()),
    // the default buffer size 8192 is too small that may impact our write
    // performance. see seastar::net::connected_socket::output()
    out(socket.output(65536)),
    socket_is_shutdown(false),
    side(_side),
    ephemeral_port(e_port)
{
  if (local_conf()->ms_tcp_nodelay) {
    socket.set_nodelay(true);
  }
}
