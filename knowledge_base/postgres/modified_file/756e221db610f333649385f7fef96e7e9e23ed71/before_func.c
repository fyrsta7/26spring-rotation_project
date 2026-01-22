		snprintf(activitymsg, sizeof(activitymsg), "last was %s", xlog);
	else
		snprintf(activitymsg, sizeof(activitymsg), "failed on %s", xlog);
	set_ps_display(activitymsg);

	return ret;
}

/*
 * pgarch_readyXlog
