	if (mm_block->prev_size != 0
		&& (prev_block=ZEND_MM_BLOCK_AT(mm_block, -(int)mm_block->prev_size))->type == ZEND_MM_FREE_BLOCK) {
		zend_mm_remove_from_free_list(heap, (zend_mm_free_block *) prev_block);
		prev_block->size += mm_block->size;
		mm_block = prev_block;
		next_block->prev_size = mm_block->size;
	}

	/* merge with the next block if empty */
	if (next_block->type == ZEND_MM_FREE_BLOCK) {
		mm_block->size += next_block->size;
		zend_mm_remove_from_free_list(heap, (zend_mm_free_block *) next_block);
		next_block = ZEND_MM_BLOCK_AT(mm_block, mm_block->size);	/* recalculate */
		next_block->prev_size = mm_block->size;
	}

	mm_block->type = ZEND_MM_FREE_BLOCK;
	zend_mm_add_to_free_list(heap, (zend_mm_free_block *) mm_block);
}

void *zend_mm_realloc(zend_mm_heap *heap, void *p, size_t size)
{
	zend_mm_block *mm_block = ZEND_MM_HEADER_OF(p);
	zend_mm_block *next_block;
	size_t true_size = MAX(ZEND_MM_ALIGNED_SIZE(size)+ZEND_MM_ALIGNED_HEADER_SIZE, ZEND_MM_ALIGNED_FREE_HEADER_SIZE);

	next_block = ZEND_MM_BLOCK_AT(mm_block, mm_block->size);

	if (true_size <= mm_block->size) {
		zend_mm_create_new_free_block(heap, mm_block, true_size);

		if (next_block->type == ZEND_MM_FREE_BLOCK) {
			zend_mm_block *new_next_block;

			new_next_block = ZEND_MM_BLOCK_AT(mm_block, mm_block->size);
			if (new_next_block != next_block) { /* A new free block was created */
				zend_mm_remove_from_free_list(heap, (zend_mm_free_block *) next_block);
				new_next_block->size += next_block->size;
				/* update the next block's prev_size */
				ZEND_MM_BLOCK_AT(mm_block, new_next_block->size)->prev_size = new_next_block->size;
			}
		}
		return p;
	}

	if ((mm_block->prev_size == 0) && (next_block->type == ZEND_MM_USED_BLOCK) &&
		(next_block->guard_block)) {
		zend_mm_segment *segment = (zend_mm_segment *) ((char *)mm_block - ZEND_MM_ALIGNED_SEGMENT_SIZE);
		zend_mm_segment *segment_copy = segment;
		zend_mm_block *guard_block;
		size_t realloc_to_size;

		/* segment size, size of block and size of guard block */
		realloc_to_size = ZEND_MM_ALIGNED_SEGMENT_SIZE+true_size+ZEND_MM_ALIGNED_HEADER_SIZE;
		segment = realloc(segment, realloc_to_size);

		if (segment != segment_copy) {
			if (heap->segments_list == segment_copy) {
				heap->segments_list = segment;
			} else {
				zend_mm_segment *seg = heap->segments_list;

				while (seg) {
					if (seg->next_segment == segment_copy) {
						seg->next_segment = segment;
						break;
					}
					seg = seg->next_segment;
				}				
			}
			mm_block = (zend_mm_block *) ((char *) segment + ZEND_MM_ALIGNED_SEGMENT_SIZE);
		}

		mm_block->size = true_size;

		/* setup guard block */
		guard_block = ZEND_MM_BLOCK_AT(mm_block, mm_block->size);
		guard_block->type = ZEND_MM_USED_BLOCK;
		guard_block->size = ZEND_MM_ALIGNED_HEADER_SIZE;
		guard_block->guard_block = 1;
		guard_block->prev_size = mm_block->size;

		return ZEND_MM_DATA_OF(mm_block);
