                                         ictx->key_down);
      if (!ictx->ar_cmd) {
	ictx->ar_state = -1;
	return NULL;
      }
      ictx->ar_state = 1;
      ictx->last_ar = t;
      return mp_cmd_clone(ictx->ar_cmd);
      // Then send rate / sec event
    } else if (ictx->ar_state == 1
               && (t -ictx->last_ar) >= 1000000 / ictx->ar_rate) {
      ictx->last_ar = t;
      return mp_cmd_clone(ictx->ar_cmd);
    }
  }
  return NULL;
}


/**
 * \param time time to wait at most for an event in milliseconds
 */
static mp_cmd_t *read_events(struct input_ctx *ictx, int time)
{
    int i;
    int got_cmd = 0;
    struct mp_input_fd *key_fds = ictx->key_fds;
    struct mp_input_fd *cmd_fds = ictx->cmd_fds;
    for (i = 0; i < ictx->num_key_fd; i++)
	if (key_fds[i].dead) {
	    mp_input_rm_key_fd(ictx, key_fds[i].fd);
	    i--;
	}
    for (i = 0; i < ictx->num_cmd_fd; i++)
	if (cmd_fds[i].dead || cmd_fds[i].eof) {
	    mp_input_rm_cmd_fd(ictx, cmd_fds[i].fd);
	    i--;
	}
	else if (cmd_fds[i].got_cmd)
	    got_cmd = 1;
#ifdef HAVE_POSIX_SELECT
    fd_set fds;
    FD_ZERO(&fds);
    if (!got_cmd) {
	int max_fd = 0;
	for (i = 0; i < ictx->num_key_fd; i++) {
	    if (key_fds[i].no_select)
		continue;
	    if (key_fds[i].fd > max_fd)
		max_fd = key_fds[i].fd;
	    FD_SET(key_fds[i].fd, &fds);
	}
	for (i = 0; i < ictx->num_cmd_fd; i++) {
	    if (cmd_fds[i].no_select)
		continue;
	    if (cmd_fds[i].fd > max_fd)
		max_fd = cmd_fds[i].fd;
	    FD_SET(cmd_fds[i].fd, &fds);
	}
        struct timeval tv, *time_val;
        if (time >= 0) {
            tv.tv_sec = time / 1000;
            tv.tv_usec = (time % 1000) * 1000;
            time_val = &tv;
        } else
            time_val = NULL;
        if (select(max_fd + 1, &fds, NULL, NULL, time_val) < 0) {
            if (errno != EINTR)
                mp_tmsg(MSGT_INPUT, MSGL_ERR, "Select error: %s\n",
                        strerror(errno));
            FD_ZERO(&fds);
        }
    }
#else
    if (!got_cmd && time)
	usec_sleep(time * 1000);
#endif


    for (i = 0; i < ictx->num_key_fd; i++) {
#ifdef HAVE_POSIX_SELECT
	if (!key_fds[i].no_select && !FD_ISSET(key_fds[i].fd, &fds))
	    continue;
#endif

	int code = key_fds[i].read_func.key(key_fds[i].ctx, key_fds[i].fd);
	if (code >= 0) {
	    mp_cmd_t *ret = interpret_key(ictx, code);
	    if (ret)
		return ret;
	}
	else if (code == MP_INPUT_ERROR)
	    mp_tmsg(MSGT_INPUT, MSGL_ERR, "Error on key input file descriptor %d\n",
		   key_fds[i].fd);
	else if (code == MP_INPUT_DEAD) {
	    mp_tmsg(MSGT_INPUT, MSGL_ERR, "Dead key input on file descriptor %d\n",
		   key_fds[i].fd);
	    key_fds[i].dead = 1;
	}
    }
    mp_cmd_t *autorepeat_cmd = check_autorepeat(ictx);
    if (autorepeat_cmd)
	return autorepeat_cmd;

    for (i = 0; i < ictx->num_cmd_fd; i++) {
