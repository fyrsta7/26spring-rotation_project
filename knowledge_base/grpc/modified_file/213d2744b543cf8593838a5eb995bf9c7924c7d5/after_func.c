      gpr_mu_unlock(&ts->mu);
      break;
    }
    GRPC_STATS_INC_EXECUTOR_QUEUE_DRAINED(&exec_ctx);
    GPR_ASSERT(grpc_closure_list_empty(ts->local_elems));
    ts->local_elems = ts->elems;
    ts->elems = (grpc_closure_list)GRPC_CLOSURE_LIST_INIT;
    gpr_mu_unlock(&ts->mu);
    if (GRPC_TRACER_ON(executor_trace)) {
      gpr_log(GPR_DEBUG, "EXECUTOR[%d]: execute", (int)(ts - g_thread_state));
    }

    run_closures(&exec_ctx, &ts->local_elems);
  }
  grpc_exec_ctx_finish(&exec_ctx);
}

static void executor_push(grpc_exec_ctx *exec_ctx, grpc_closure *closure,
                          grpc_error *error, bool is_short) {
  bool retry_push;
  if (is_short) {
    GRPC_STATS_INC_EXECUTOR_SCHEDULED_SHORT_ITEMS(exec_ctx);
  } else {
    GRPC_STATS_INC_EXECUTOR_SCHEDULED_LONG_ITEMS(exec_ctx);
  }
  do {
    retry_push = false;
    size_t cur_thread_count = (size_t)gpr_atm_no_barrier_load(&g_cur_threads);
    if (cur_thread_count == 0) {
      if (GRPC_TRACER_ON(executor_trace)) {
#ifndef NDEBUG
        gpr_log(GPR_DEBUG, "EXECUTOR: schedule %p (created %s:%d) inline",
                closure, closure->file_created, closure->line_created);
#else
        gpr_log(GPR_DEBUG, "EXECUTOR: schedule %p inline", closure);
#endif
      }
      grpc_closure_list_append(&exec_ctx->closure_list, closure, error);
      return;
    }
    thread_state *ts = (thread_state *)gpr_tls_get(&g_this_thread_state);
    if (ts == NULL) {
      ts = &g_thread_state[GPR_HASH_POINTER(exec_ctx, cur_thread_count)];
    } else {
      GRPC_STATS_INC_EXECUTOR_SCHEDULED_TO_SELF(exec_ctx);
      if (!is_short) {
        grpc_closure_list_append(&ts->local_elems, closure, error);
        return;
      }
    }
    thread_state *orig_ts = ts;

    bool try_new_thread;
    for (;;) {
      if (GRPC_TRACER_ON(executor_trace)) {
#ifndef NDEBUG
        gpr_log(
            GPR_DEBUG,
            "EXECUTOR: try to schedule %p (%s) (created %s:%d) to thread %d",
            closure, is_short ? "short" : "long", closure->file_created,
            closure->line_created, (int)(ts - g_thread_state));
#else
        gpr_log(GPR_DEBUG, "EXECUTOR: try to schedule %p (%s) to thread %d",
                closure, is_short ? "short" : "long",
                (int)(ts - g_thread_state));
#endif
      }
      gpr_mu_lock(&ts->mu);
      if (ts->queued_long_job) {
        // if there's a long job queued, we never queue anything else to this
        // queue (since long jobs can take 'infinite' time and we need to
        // guarantee no starvation)
        // ... spin through queues and try again
        gpr_mu_unlock(&ts->mu);
        size_t idx = (size_t)(ts - g_thread_state);
        ts = &g_thread_state[(idx + 1) % cur_thread_count];
        if (ts == orig_ts) {
          retry_push = true;
          try_new_thread = true;
          break;
        }
        continue;
      }
      if (grpc_closure_list_empty(ts->elems)) {
        GRPC_STATS_INC_EXECUTOR_WAKEUP_INITIATED(exec_ctx);
        gpr_cv_signal(&ts->cv);
      }
      grpc_closure_list_append(&ts->elems, closure, error);
      try_new_thread = ts->elems.head != closure &&
                       cur_thread_count < g_max_threads && !ts->shutdown;
      if (!is_short) ts->queued_long_job = true;
      gpr_mu_unlock(&ts->mu);
      break;
    }
