  }

  buf_free(buf);

  nrecv += bytes;
  nrecv_total += bytes;
}


static void write_cb(uv_write_t* req, int status) {
  ASSERT(status == 0);

