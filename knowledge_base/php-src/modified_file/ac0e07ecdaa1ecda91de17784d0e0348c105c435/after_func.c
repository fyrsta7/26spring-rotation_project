	return p;
}


ZEND_API int zend_set_memory_limit(unsigned int memory_limit)
{
#if MEMORY_LIMIT
	ALS_FETCH();

	AG(memory_limit) = memory_limit;
	return SUCCESS;
#else
	return FAILURE;
#endif
}


ZEND_API void start_memory_manager(ALS_D)
{
	int i, j;
	void *cached_entries[MAX_CACHED_MEMORY][MAX_CACHED_ENTRIES];

	AG(phead) = AG(head) = NULL;
	
#if MEMORY_LIMIT
	AG(memory_limit)=1<<30;		/* rediculous limit, effectively no limit */
	AG(allocated_memory)=0;
	AG(memory_exhausted)=0;
#endif

#if ZEND_DEBUG
	memset(AG(cache_stats), 0, sizeof(AG(cache_stats)));
	memset(AG(fast_cache_stats), 0, sizeof(AG(fast_cache_stats)));
