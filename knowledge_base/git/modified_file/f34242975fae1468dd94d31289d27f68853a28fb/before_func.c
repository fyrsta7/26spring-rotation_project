
		/*
		 * Reordering the records might have moved a short one
		 * to the end of the buffer, so verify the buffer's
		 * safety again:
		 */
		verify_buffer_safe(snapshot);
	}

	if (mmap_strategy != MMAP_OK && snapshot->mmapped) {
		/*
		 * We don't want to leave the file mmapped, so we are
		 * forced to make a copy now:
		 */
		size_t size = snapshot->eof - snapshot->start;
		char *buf_copy = xmalloc(size);

		memcpy(buf_copy, snapshot->start, size);
		clear_snapshot_buffer(snapshot);
		snapshot->buf = snapshot->start = buf_copy;
		snapshot->eof = buf_copy + size;
	}

	return snapshot;
}

/*
 * Check that `refs->snapshot` (if present) still reflects the
 * contents of the `packed-refs` file. If not, clear the snapshot.
 */
static void validate_snapshot(struct packed_ref_store *refs)
{
	if (refs->snapshot &&
	    !stat_validity_check(&refs->snapshot->validity, refs->path))
		clear_snapshot(refs);
}

/*
 * Get the `snapshot` for the specified packed_ref_store, creating and
 * populating it if it hasn't been read before or if the file has been
 * changed (according to its `validity` field) since it was last read.
 * On the other hand, if we hold the lock, then assume that the file
 * hasn't been changed out from under us, so skip the extra `stat()`
 * call in `stat_validity_check()`. This function does *not* increase
 * the snapshot's reference count on behalf of the caller.
 */
static struct snapshot *get_snapshot(struct packed_ref_store *refs)
