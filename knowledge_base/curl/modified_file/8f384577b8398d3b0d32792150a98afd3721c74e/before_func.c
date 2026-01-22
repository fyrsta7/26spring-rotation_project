  if(len && (new_path[len - 1] == '\"')) {
    new_path[len - 1] = 0x0;
    len--;
  }

  /* RFC6265 5.2.4 The Path Attribute */
  if(new_path[0] != '/') {
    /* Let cookie-path be the default-path. */
    strstore(&new_path, "/");
    return new_path;
  }

  /* convert /hoge/ to /hoge */
  if(len && new_path[len - 1] == '/') {
