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
