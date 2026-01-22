}

static CURLcode ng_process_ingress(struct Curl_easy *data,
                                   curl_socket_t sockfd,
                                   struct quicsocket *qs)
{
  ssize_t recvd;
  int rv;
  uint8_t buf[65536];
  size_t bufsize = sizeof(buf);
  struct sockaddr_storage remote_addr;
  socklen_t remote_addrlen;
  ngtcp2_path path;
  ngtcp2_tstamp ts = timestamp();
  ngtcp2_pkt_info pi = { 0 };

  for(;;) {
    remote_addrlen = sizeof(remote_addr);
    while((recvd = recvfrom(sockfd, (char *)buf, bufsize, 0,
                            (struct sockaddr *)&remote_addr,
                            &remote_addrlen)) == -1 &&
          SOCKERRNO == EINTR)
      ;
    if(recvd == -1) {
      if(SOCKERRNO == EAGAIN || SOCKERRNO == EWOULDBLOCK)
        break;

      failf(data, "ngtcp2: recvfrom() unexpectedly returned %zd", recvd);
      return CURLE_RECV_ERROR;
    }

    ngtcp2_addr_init(&path.local, (struct sockaddr *)&qs->local_addr,
                     qs->local_addrlen);
    ngtcp2_addr_init(&path.remote, (struct sockaddr *)&remote_addr,
                     remote_addrlen);

    rv = ngtcp2_conn_read_pkt(qs->qconn, &path, &pi, buf, recvd, ts);
    if(rv) {
      /* TODO Send CONNECTION_CLOSE if possible */
      if(rv == NGTCP2_ERR_CRYPTO)
        /* this is a "TLS problem", but a failed certificate verification
           is a common reason for this */
        return CURLE_PEER_FAILED_VERIFICATION;
      return CURLE_RECV_ERROR;
    }
  }

  return CURLE_OK;
}

static CURLcode ng_flush_egress(struct Curl_easy *data,
                                int sockfd,
                                struct quicsocket *qs)
{
  int rv;
  ssize_t sent;
  ssize_t outlen;
  uint8_t out[NGTCP2_MAX_UDP_PAYLOAD_SIZE];
  ngtcp2_path_storage ps;
  ngtcp2_tstamp ts = timestamp();
  struct sockaddr_storage remote_addr;
  ngtcp2_tstamp expiry;
  ngtcp2_duration timeout;
  int64_t stream_id;
  ssize_t veccnt;
  int fin;
  nghttp3_vec vec[16];
  ssize_t ndatalen;
  uint32_t flags;

  rv = ngtcp2_conn_handle_expiry(qs->qconn, ts);
  if(rv) {
    failf(data, "ngtcp2_conn_handle_expiry returned error: %s",
          ngtcp2_strerror(rv));
    return CURLE_SEND_ERROR;
  }

  ngtcp2_path_storage_zero(&ps);

  for(;;) {
    veccnt = 0;
    stream_id = -1;
    fin = 0;

    if(qs->h3conn && ngtcp2_conn_get_max_data_left(qs->qconn)) {
      veccnt = nghttp3_conn_writev_stream(qs->h3conn, &stream_id, &fin, vec,
                                          sizeof(vec) / sizeof(vec[0]));
      if(veccnt < 0) {
        failf(data, "nghttp3_conn_writev_stream returned error: %s",
              nghttp3_strerror((int)veccnt));
        return CURLE_SEND_ERROR;
      }
    }

    flags = NGTCP2_WRITE_STREAM_FLAG_MORE |
            (fin ? NGTCP2_WRITE_STREAM_FLAG_FIN : 0);
    outlen = ngtcp2_conn_writev_stream(qs->qconn, &ps.path, NULL, out,
                                       sizeof(out),
                                       &ndatalen, flags, stream_id,
                                       (const ngtcp2_vec *)vec, veccnt, ts);
    if(outlen == 0) {
      break;
    }
    if(outlen < 0) {
      switch(outlen) {
      case NGTCP2_ERR_STREAM_DATA_BLOCKED:
        assert(ndatalen == -1);
        rv = nghttp3_conn_block_stream(qs->h3conn, stream_id);
        if(rv) {
          failf(data, "nghttp3_conn_block_stream returned error: %s\n",
                nghttp3_strerror(rv));
          return CURLE_SEND_ERROR;
        }
        continue;
      case NGTCP2_ERR_STREAM_SHUT_WR:
        assert(ndatalen == -1);
        rv = nghttp3_conn_shutdown_stream_write(qs->h3conn, stream_id);
        if(rv) {
          failf(data,
                "nghttp3_conn_shutdown_stream_write returned error: %s\n",
                nghttp3_strerror(rv));
          return CURLE_SEND_ERROR;
        }
        continue;
      case NGTCP2_ERR_WRITE_MORE:
        assert(ndatalen >= 0);
        rv = nghttp3_conn_add_write_offset(qs->h3conn, stream_id, ndatalen);
        if(rv) {
          failf(data, "nghttp3_conn_add_write_offset returned error: %s\n",
                nghttp3_strerror(rv));
