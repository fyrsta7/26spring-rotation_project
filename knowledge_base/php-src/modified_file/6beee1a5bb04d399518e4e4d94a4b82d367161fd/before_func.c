		RETURN_THROWS();
	}

	if (wouldblock) {
		ZEND_TRY_ASSIGN_REF_LONG(wouldblock, 0);
	}

	/* flock_values contains all possible actions if (operation & PHP_LOCK_NB) we won't block on the lock */
	act = flock_values[act - 1] | (operation & PHP_LOCK_NB ? LOCK_NB : 0);
	if (php_stream_lock(stream, act)) {
		if (operation && errno == EWOULDBLOCK && wouldblock) {
			ZEND_TRY_ASSIGN_REF_LONG(wouldblock, 1);
		}
		RETURN_FALSE;
	}
	RETURN_TRUE;
}
