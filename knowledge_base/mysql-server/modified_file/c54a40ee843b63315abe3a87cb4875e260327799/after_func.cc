    src+= srcres;
    dst+= dstres;
  }
  *dst= '\0';
  return (size_t) (dst - dst0);
}


static size_t
my_casedn_utf8mb4(const CHARSET_INFO *cs,
                  char *src, size_t srclen,
                  char *dst, size_t dstlen)
{
  my_wc_t wc;
  int srcres, dstres;
  char *srcend= src + srclen, *dstend= dst + dstlen, *dst0= dst;
  const MY_UNICASE_INFO *uni_plane= cs->caseinfo;
  DBUG_ASSERT(src != dst || cs->casedn_multiply == 1);

  while ((src < srcend) &&
         (srcres= my_mb_wc_utf8mb4(&wc, (uchar*) src, (uchar*) srcend)) > 0)
  {
