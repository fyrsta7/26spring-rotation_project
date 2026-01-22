  struct priv *p = s->priv;
  int r;
  int wr = 0;
  while (wr < len) {
    r = smbc_write(p->fd,buffer,len);
    if (r <= 0)
      return -1;
    wr += r;
    buffer += r;
  }
  return len;
}

static void close_f(stream_t *s){
  struct priv *p = s->priv;
  smbc_close(p->fd);
}

static int open_f (stream_t *stream)
{
  char *filename;
  int64_t len;
  int fd, err;

  struct priv *priv = talloc_zero(stream, struct priv);
  stream->priv = priv;

  filename = stream->url;

  bool write = stream->mode == STREAM_WRITE;
  mode_t m = write ? O_RDWR|O_CREAT|O_TRUNC : O_RDONLY;

  if(!filename) {
    MP_ERR(stream, "[smb] Bad url\n");
    return STREAM_ERROR;
  }

  err = smbc_init(smb_auth_fn, 1);
  if (err < 0) {
    MP_ERR(stream, "Cannot init the libsmbclient library: %d\n",err);
    return STREAM_ERROR;
  }

  fd = smbc_open(filename, m,0644);
  if (fd < 0) {
    MP_ERR(stream, "Could not open from LAN: '%s'\n", filename);
    return STREAM_ERROR;
  }

  len = 0;
