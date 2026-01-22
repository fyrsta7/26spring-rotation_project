#endif

static void error_destroy(grpc_error *err) {
  GPR_ASSERT(!grpc_error_is_special(err));
  gpr_avl_unref(err->ints);
  gpr_avl_unref(err->strs);
  gpr_avl_unref(err->errs);
  gpr_avl_unref(err->times);
  gpr_free((void *)gpr_atm_acq_load(&err->error_string));
  gpr_free(err);
}

#ifdef GRPC_ERROR_REFCOUNT_DEBUG
void grpc_error_unref(grpc_error *err, const char *file, int line,
                      const char *func) {
  if (grpc_error_is_special(err)) return;
  gpr_log(GPR_DEBUG, "%p: %" PRIdPTR " -> %" PRIdPTR " [%s:%d %s]", err,
          err->refs.count, err->refs.count - 1, file, line, func);
  if (gpr_unref(&err->refs)) {
    error_destroy(err);
  }
}
#else
void grpc_error_unref(grpc_error *err) {
  if (grpc_error_is_special(err)) return;
  if (gpr_unref(&err->refs)) {
    error_destroy(err);
  }
}
#endif

grpc_error *grpc_error_create(const char *file, int line, const char *desc,
                              grpc_error **referencing,
