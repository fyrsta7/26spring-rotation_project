  char *dst= src, *dst0= src;
  const MY_UNICASE_INFO *uni_plane= cs->caseinfo;
  DBUG_ASSERT(cs->caseup_multiply == 1);

  while (*src &&
         (srcres= my_mb_wc_utf8mb4_no_range(cs, &wc, (uchar *) src)) > 0)
  {
    my_toupper_utf8mb4(uni_plane, &wc);
