		snprintf(activitymsg, sizeof(activitymsg), "last was %s", xlog);
	else
		snprintf(activitymsg, sizeof(activitymsg), "failed on %s", xlog);
	set_ps_display(activitymsg);

	return ret;
}

/*
 * pgarch_readyXlog
 *
 * Return name of the oldest xlog file that has not yet been archived.
 * No notification is set that file archiving is now in progress, so
 * this would need to be extended if multiple concurrent archival
 * tasks were created. If a failure occurs, we will completely
 * re-copy the file at the next available opportunity.
