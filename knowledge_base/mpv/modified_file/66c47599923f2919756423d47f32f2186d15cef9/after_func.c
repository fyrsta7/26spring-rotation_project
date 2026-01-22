  float *pw, *po, *ppc, *search_start;
  float best_corr = INT_MIN;
  int best_off = 0;
  int i, off;

  pw  = s->table_window;
  po  = s->buf_overlap;
  po += s->num_channels;
  ppc = s->buf_pre_corr;
  for (i=s->num_channels; i<s->samples_overlap; i++) {
    *ppc++ = *pw++ * *po++;
  }

  search_start = (float*)s->buf_queue + s->num_channels;
  for (off=0; off<s->frames_search; off++) {
    float corr = 0;
    float* ps = search_start;
    ppc = s->buf_pre_corr;
    for (i=s->num_channels; i<s->samples_overlap; i++) {
      corr += *ppc++ * *ps++;
    }
    if (corr > best_corr) {
      best_corr = corr;
      best_off  = off;
    }
    search_start += s->num_channels;
  }

  return best_off * 4 * s->num_channels;
}

static int best_overlap_offset_s16(af_scaletempo_t* s)
{
  int32_t *pw, *ppc;
  int16_t *po, *search_start;
  int32_t best_corr = INT_MIN;
  int best_off = 0;
