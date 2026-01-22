  GPR_UNREACHABLE_CODE(return "bad state tuple");
}

static void write_action_begin_locked(void* gt, grpc_error* error_ignored) {
  GPR_TIMER_SCOPE("write_action_begin_locked", 0);
  grpc_chttp2_transport* t = static_cast<grpc_chttp2_transport*>(gt);
  GPR_ASSERT(t->write_state != GRPC_CHTTP2_WRITE_STATE_IDLE);
  grpc_chttp2_begin_write_result r;
  if (t->closed_with_error != GRPC_ERROR_NONE) {
    r.writing = false;
  } else {
    r = grpc_chttp2_begin_write(t);
  }
  if (r.writing) {
    if (r.partial) {
      GRPC_STATS_INC_HTTP2_PARTIAL_WRITES();
    }
    if (!t->is_first_write_in_batch) {
      GRPC_STATS_INC_HTTP2_WRITES_CONTINUED();
    }
    grpc_closure_scheduler* scheduler =
        write_scheduler(t, r.early_results_scheduled, r.partial);
    if (scheduler != grpc_schedule_on_exec_ctx) {
      GRPC_STATS_INC_HTTP2_WRITES_OFFLOADED();
    }
    set_write_state(
        t,
        r.partial ? GRPC_CHTTP2_WRITE_STATE_WRITING_WITH_MORE
                  : GRPC_CHTTP2_WRITE_STATE_WRITING,
        begin_writing_desc(r.partial, scheduler == grpc_schedule_on_exec_ctx));
    GRPC_CLOSURE_SCHED(
        GRPC_CLOSURE_INIT(&t->write_action, write_action, t, scheduler),
        GRPC_ERROR_NONE);
  } else {
    GRPC_STATS_INC_HTTP2_SPURIOUS_WRITES_BEGUN();
    set_write_state(t, GRPC_CHTTP2_WRITE_STATE_IDLE, "begin writing nothing");
    GRPC_CHTTP2_UNREF_TRANSPORT(t, "writing");
  }
}

