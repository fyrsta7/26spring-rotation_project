    if (ret != Z_OK)
      return false;
    if (strm.avail_out == 0)
    {
     dest_buffer.append(tmp_buf, BUFSIZE);
     strm.next_out = reinterpret_cast<unsigned char*>(tmp_buf);
     strm.avail_out = BUFSIZE;
    }
   }

  int deflate_res = Z_OK;
  while (deflate_res == Z_OK) {
    if (strm.avail_out == 0) {
      dest_buffer.append(tmp_buf, BUFSIZE);
      strm.next_out = reinterpret_cast<unsigned char*>(tmp_buf);
      strm.avail_out = BUFSIZE;
