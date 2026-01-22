 * Check the validity of the parameters.
 * Returns non-zero if the parameters are valid and 0 otherwise.
 */
static int COVER_checkParameters(ZDICT_cover_params_t parameters,
                                 size_t maxDictSize) {
  /* k and d are required parameters */
  if (parameters.d == 0 || parameters.k == 0) {
    return 0;
  }
  /* k <= maxDictSize */
  if (parameters.k > maxDictSize) {
    return 0;
  }
  /* d <= k */
  if (parameters.d > parameters.k) {
    return 0;
  }
  return 1;
}

/**
 * Clean up a context initialized with `COVER_ctx_init()`.
 */
static void COVER_ctx_destroy(COVER_ctx_t *ctx) {
  if (!ctx) {
    return;
  }
  if (ctx->suffix) {
    free(ctx->suffix);
    ctx->suffix = NULL;
  }
  if (ctx->freqs) {
    free(ctx->freqs);
    ctx->freqs = NULL;
  }
  if (ctx->dmerAt) {
    free(ctx->dmerAt);
    ctx->dmerAt = NULL;
  }
