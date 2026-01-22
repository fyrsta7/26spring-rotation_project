os_mutex_exit(
/*==========*/
	os_mutex_t	mutex)	/* in: mutex to release */
{
#ifdef __WIN__
	ut_a(mutex);

	ut_a(mutex->count == 1);

	(mutex->count)--;

	ut_a(ReleaseMutex(mutex->handle));
#else
