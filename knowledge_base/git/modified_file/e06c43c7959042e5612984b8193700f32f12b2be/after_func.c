
		src_offset += ondisk_ce_size(ce);
		dst_offset += ce_size(ce);
	}
	istate->timestamp = st.st_mtime;
	while (src_offset <= mmap_size - 20 - 8) {
		/* After an array of active_nr index entries,
		 * there can be arbitrary number of extended
		 * sections, each of which is prefixed with
		 * extension name (4-byte) and section length
		 * in 4-byte network byte order.
		 */
		unsigned long extsize;
		memcpy(&extsize, (char *)mmap + src_offset + 4, 4);
		extsize = ntohl(extsize);
		if (read_index_extension(istate,
					 (const char *) mmap + src_offset,
					 (char *) mmap + src_offset + 8,
					 extsize) < 0)
			goto unmap;
		src_offset += 8;
		src_offset += extsize;
	}
	munmap(mmap, mmap_size);
	return istate->cache_nr;

unmap:
	munmap(mmap, mmap_size);
	errno = EINVAL;
	die("index file corrupt");
}

int discard_index(struct index_state *istate)
{
	istate->cache_nr = 0;
	istate->cache_changed = 0;
	istate->timestamp = 0;
	free_hash(&istate->name_hash);
	cache_tree_free(&(istate->cache_tree));
	free(istate->alloc);
	istate->alloc = NULL;

	/* no need to throw away allocated active_cache */
	return 0;
