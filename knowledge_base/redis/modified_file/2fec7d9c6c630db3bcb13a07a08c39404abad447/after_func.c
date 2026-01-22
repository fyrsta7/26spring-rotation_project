
void
background_thread_postfork_child(tsdn_t *tsdn) {
	for (unsigned i = 0; i < max_background_threads; i++) {
		malloc_mutex_postfork_child(tsdn,
		    &background_thread_info[i].mtx);
	}
	malloc_mutex_postfork_child(tsdn, &background_thread_lock);
	if (!background_thread_enabled_at_fork) {
		return;
	}

	/* Clear background_thread state (reset to disabled for child). */
	malloc_mutex_lock(tsdn, &background_thread_lock);
	n_background_threads = 0;
	background_thread_enabled_set(tsdn, false);
	for (unsigned i = 0; i < max_background_threads; i++) {
		background_thread_info_t *info = &background_thread_info[i];
		malloc_mutex_lock(tsdn, &info->mtx);
		info->state = background_thread_stopped;
		int ret = pthread_cond_init(&info->cond, NULL);
		assert(ret == 0);
		background_thread_info_init(tsdn, info);
		malloc_mutex_unlock(tsdn, &info->mtx);
	}
	malloc_mutex_unlock(tsdn, &background_thread_lock);
}

bool
background_thread_stats_read(tsdn_t *tsdn, background_thread_stats_t *stats) {
	assert(config_stats);
	malloc_mutex_lock(tsdn, &background_thread_lock);
