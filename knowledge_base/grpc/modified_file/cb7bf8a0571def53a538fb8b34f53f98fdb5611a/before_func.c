  pollset->in_flight_cbs = 0;
  pollset->shutting_down = 0;
  pollset->called_shutdown = 0;
  pollset->idle_jobs.head = pollset->idle_jobs.tail = NULL;
  become_basic_pollset(pollset, NULL);
}

void grpc_pollset_add_fd(grpc_exec_ctx *exec_ctx, grpc_pollset *pollset,
                         grpc_fd *fd) {
  gpr_mu_lock(&pollset->mu);
  pollset->vtable->add_fd(exec_ctx, pollset, fd, 1);
/* the following (enabled only in debug) will reacquire and then release
   our lock - meaning that if the unlocking flag passed to del_fd above is
   not respected, the code will deadlock (in a way that we have a chance of
   debugging) */
#ifndef NDEBUG
  gpr_mu_lock(&pollset->mu);
  gpr_mu_unlock(&pollset->mu);
#endif
}

void grpc_pollset_del_fd(grpc_exec_ctx *exec_ctx, grpc_pollset *pollset,
                         grpc_fd *fd) {
  gpr_mu_lock(&pollset->mu);
  pollset->vtable->del_fd(exec_ctx, pollset, fd, 1);
/* the following (enabled only in debug) will reacquire and then release
   our lock - meaning that if the unlocking flag passed to del_fd above is
   not respected, the code will deadlock (in a way that we have a chance of
   debugging) */
#ifndef NDEBUG
  gpr_mu_lock(&pollset->mu);
  gpr_mu_unlock(&pollset->mu);
#endif
}

static void finish_shutdown(grpc_exec_ctx *exec_ctx, grpc_pollset *pollset) {
  GPR_ASSERT(grpc_closure_list_empty(pollset->idle_jobs));
  pollset->vtable->finish_shutdown(pollset);
  grpc_exec_ctx_enqueue(exec_ctx, pollset->shutdown_done, 1);
}

void grpc_pollset_work(grpc_exec_ctx *exec_ctx, grpc_pollset *pollset,
                       grpc_pollset_worker *worker, gpr_timespec now,
                       gpr_timespec deadline) {
  /* pollset->mu already held */
  int added_worker = 0;
  int locked = 1;
  int queued_work = 0;
  int keep_polling = 0;
  GRPC_TIMER_BEGIN("grpc_pollset_work", 0);
  /* this must happen before we (potentially) drop pollset->mu */
  worker->next = worker->prev = NULL;
  worker->reevaluate_polling_on_wakeup = 0;
  /* TODO(ctiller): pool these */
  grpc_wakeup_fd_init(&worker->wakeup_fd);
  /* If there's work waiting for the pollset to be idle, and the
     pollset is idle, then do that work */
  if (!grpc_pollset_has_workers(pollset) &&
      !grpc_closure_list_empty(pollset->idle_jobs)) {
    grpc_exec_ctx_enqueue_list(exec_ctx, &pollset->idle_jobs);
    goto done;
  }
  /* Check alarms - these are a global resource so we just ping
     each time through on every pollset.
     May update deadline to ensure timely wakeups.
     TODO(ctiller): can this work be localized? */
  if (grpc_alarm_check(exec_ctx, now, &deadline)) {
    gpr_mu_unlock(&pollset->mu);
    locked = 0;
    goto done;
  }
  /* If we're shutting down then we don't execute any extended work */
  if (pollset->shutting_down) {
    goto done;
  }
  /* Give do_promote priority so we don't starve it out */
  if (pollset->in_flight_cbs) {
    gpr_mu_unlock(&pollset->mu);
    locked = 0;
    goto done;
  }
  /* Start polling, and keep doing so while we're being asked to
     re-evaluate our pollers (this allows poll() based pollers to
     ensure they don't miss wakeups) */
  keep_polling = 1;
  while (keep_polling) {
    keep_polling = 0;
    if (!pollset->kicked_without_pollers) {
      if (!added_worker) {
        push_front_worker(pollset, worker);
        added_worker = 1;
      }
      gpr_tls_set(&g_current_thread_poller, (gpr_intptr)pollset);
      gpr_tls_set(&g_current_thread_worker, (gpr_intptr)worker);
      GRPC_TIMER_BEGIN("maybe_work_and_unlock", 0);
      pollset->vtable->maybe_work_and_unlock(exec_ctx, pollset, worker,
                                             deadline, now);
      GRPC_TIMER_END("maybe_work_and_unlock", 0);
