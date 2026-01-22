bool BufferManager::EvictBlocks(idx_t extra_memory, idx_t memory_limit) {
	PurgeQueue();
#ifdef DEBUG
	VerifyCurrentMemory();
#endif
	current_memory += extra_memory;
	if (!(current_memory > 0.9 * memory_limit && io_lock.try_lock())) {
		// we did not get the IO lock
		if (current_memory < memory_limit) {
			// memory is not full, yay!
			return true;
		} else {
			// memory is full, wait until there is space
			lock_guard<mutex> wait_until_space(memory_full_lock);
			return true;
		}
	}

	// we got the IO lock, unload until we have some room
	vector<shared_ptr<BlockHandle>> handles_to_unload;
	idx_t memory_to_free = 0;

	bool mf_locked = false;
	unique_ptr<BufferEvictionNode> node;
	while (current_memory - memory_to_free > 0.9 * memory_limit) {
		// lock so the other threads have to wait until there is space
		if (!mf_locked && current_memory > memory_limit) {
			memory_full_lock.lock();
			mf_locked = true;
		}

		// get a block to unpin from the queue
		if (!queue->q.try_dequeue(node)) {
			// we cannot unload
			current_memory -= extra_memory;
			break;
		}
		// get a reference to the underlying block pointer
		auto handle = node->TryGetBlockHandle();
		if (!handle) {
			continue;
		}
		// we might be able to free this block: grab the mutex and check if we can free it
		handle->lock.lock();
		if (!node->CanUnload(*handle)) {
			// something changed in the mean-time, bail out
			handle->lock.unlock();
			continue;
		}
		// hooray, we can unload the block
		// add the block to the blocks that will be unloaded
		memory_to_free += handle->memory_usage;
		handles_to_unload.push_back(move(handle));
		// if we can unload 5% of memory, do it
		if (memory_to_free >= 0.05 * memory_limit) {
			for (auto &h : handles_to_unload) {
				h->Unload();
			}
			for (auto &h : handles_to_unload) {
				h->lock.unlock();
			}
			handles_to_unload.clear();
			if (mf_locked && current_memory <= memory_limit) {
				memory_full_lock.unlock();
				mf_locked = false;
			}
		}
	}
	// unload remaining blocks
	for (auto &h : handles_to_unload) {
		h->Unload();
	}
	for (auto &h : handles_to_unload) {
		h->lock.unlock();
	}
	// unlock io lock again
	io_lock.unlock();
	if (mf_locked) {
		// we could not free up enough space
		memory_full_lock.unlock();
		return false;
	} else {
		// we freed up enough space!
		return true;
	}
}
