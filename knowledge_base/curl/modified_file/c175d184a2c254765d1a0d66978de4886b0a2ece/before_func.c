      if(result == CURLE_AGAIN) {
        *err = result;
        return -1;
      }

      if(nread == -1) {
        failf(data, "Failed receiving HTTP2 data");
        *err = result;
        return 0;
      }

      if(nread == 0) {
        failf(data, "Unexpected EOF");
        *err = CURLE_RECV_ERROR;
        return -1;
      }

      DEBUGF(infof(data, "nread=%zd\n", nread));

      httpc->inbuflen = nread;
      inbuf = httpc->inbuf;
    }
    else {
      nread = httpc->inbuflen - httpc->nread_inbuf;
      inbuf = httpc->inbuf + httpc->nread_inbuf;

      DEBUGF(infof(data, "Use data left in connection buffer, nread=%zd\n",
                   nread));
    }
    rv = nghttp2_session_mem_recv(httpc->h2, (const uint8_t *)inbuf, nread);

    if(nghttp2_is_fatal((int)rv)) {
      failf(data, "nghttp2_session_mem_recv() returned %d:%s\n",
            rv, nghttp2_strerror((int)rv));
      *err = CURLE_RECV_ERROR;
      return 0;
    }
    DEBUGF(infof(data, "nghttp2_session_mem_recv() returns %zd\n", rv));
    if(nread == rv) {
      DEBUGF(infof(data, "All data in connection buffer processed\n"));
      httpc->inbuflen = 0;
      httpc->nread_inbuf = 0;
    }
    else {
      httpc->nread_inbuf += rv;
      DEBUGF(infof(data, "%zu bytes left in connection buffer\n",
                   httpc->inbuflen - httpc->nread_inbuf));
    }
    /* Always send pending frames in nghttp2 session, because
       nghttp2_session_mem_recv() may queue new frame */
    rv = nghttp2_session_send(httpc->h2);
    if(rv != 0) {
      *err = CURLE_SEND_ERROR;
      return 0;
    }
  }
  if(stream->memlen) {
    ssize_t retlen = stream->memlen;
    infof(data, "http2_recv: returns %zd for stream %x\n",
          retlen, stream->stream_id);
    stream->memlen = 0;

    if(httpc->pause_stream_id == stream->stream_id) {
      /* data for this stream is returned now, but this stream caused a pause
         already so we need it called again asap */
      DEBUGF(infof(data, "Data returned for PAUSED stream %x\n",
                   stream->stream_id));
    }
    else
      data->state.drain = 0; /* this stream is hereby drained */

    return retlen;
  }
  /* If stream is closed, return 0 to signal the http routine to close
     the connection */
  if(stream->closed) {
    return http2_handle_stream_close(httpc, data, stream, err);
  }
  *err = CURLE_AGAIN;
  DEBUGF(infof(data, "http2_recv returns AGAIN for stream %x\n",
               stream->stream_id));
  return -1;
}

/* Index where :authority header field will appear in request header
   field list. */
#define AUTHORITY_DST_IDX 3

/* return number of received (decrypted) bytes */
static ssize_t http2_send(struct connectdata *conn, int sockindex,
                          const void *mem, size_t len, CURLcode *err)
{
  /*
   * BIG TODO: Currently, we send request in this function, but this
   * function is also used to send request body. It would be nice to
   * add dedicated function for request.
   */
  int rv;
  struct http_conn *httpc = &conn->proto.httpc;
  struct HTTP *stream = conn->data->req.protop;
  nghttp2_nv *nva;
  size_t nheader;
  size_t i;
  size_t authority_idx;
  char *hdbuf = (char*)mem;
  char *end;
  nghttp2_data_provider data_prd;
  int32_t stream_id;
  nghttp2_session *h2 = httpc->h2;

  (void)sockindex;

  DEBUGF(infof(conn->data, "http2_send len=%zu\n", len));

  if(stream->stream_id != -1) {
    /* If stream_id != -1, we have dispatched request HEADERS, and now
       are going to send or sending request body in DATA frame */
    stream->upload_mem = mem;
    stream->upload_len = len;
    nghttp2_session_resume_data(h2, stream->stream_id);
    rv = nghttp2_session_send(h2);
    if(nghttp2_is_fatal(rv)) {
      *err = CURLE_SEND_ERROR;
      return -1;
    }
    len -= stream->upload_len;

    DEBUGF(infof(conn->data, "http2_send returns %zu for stream %x\n", len,
                 stream->stream_id));
    return len;
  }

  /* Calculate number of headers contained in [mem, mem + len) */
  /* Here, we assume the curl http code generate *correct* HTTP header
     field block */
  nheader = 0;
  for(i = 0; i < len; ++i) {
    if(hdbuf[i] == 0x0a) {
      ++nheader;
    }
  }
  /* We counted additional 2 \n in the first and last line. We need 3
     new headers: :method, :path and :scheme. Therefore we need one
     more space. */
  nheader += 1;
  nva = malloc(sizeof(nghttp2_nv) * nheader);
  if(nva == NULL) {
    *err = CURLE_OUT_OF_MEMORY;
    return -1;
  }
  /* Extract :method, :path from request line */
  end = strchr(hdbuf, ' ');
  nva[0].name = (unsigned char *)":method";
  nva[0].namelen = (uint16_t)strlen((char *)nva[0].name);
  nva[0].value = (unsigned char *)hdbuf;
  nva[0].valuelen = (uint16_t)(end - hdbuf);
  nva[0].flags = NGHTTP2_NV_FLAG_NONE;

  hdbuf = end + 1;

  end = strchr(hdbuf, ' ');
  nva[1].name = (unsigned char *)":path";
  nva[1].namelen = (uint16_t)strlen((char *)nva[1].name);
  nva[1].value = (unsigned char *)hdbuf;
  nva[1].valuelen = (uint16_t)(end - hdbuf);
  nva[1].flags = NGHTTP2_NV_FLAG_NONE;

  nva[2].name = (unsigned char *)":scheme";
  nva[2].namelen = (uint16_t)strlen((char *)nva[2].name);
  if(conn->handler->flags & PROTOPT_SSL)
    nva[2].value = (unsigned char *)"https";
  else
    nva[2].value = (unsigned char *)"http";
  nva[2].valuelen = (uint16_t)strlen((char *)nva[2].value);
  nva[2].flags = NGHTTP2_NV_FLAG_NONE;

  hdbuf = strchr(hdbuf, 0x0a);
  ++hdbuf;
