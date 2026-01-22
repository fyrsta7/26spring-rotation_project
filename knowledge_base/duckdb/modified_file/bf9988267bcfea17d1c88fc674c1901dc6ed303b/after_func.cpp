bool BufferedCSVReader::ReadBuffer(idx_t &start) {
	auto old_buffer = move(buffer);
	idx_t read_count;
	// the remaining part of the last buffer
	idx_t remaining = buffer_size - start;

	if (ra_fetch_in_progress) {
		std::unique_lock<std::mutex> lck(ra_fetch_lock);
		ra_fetch_cv.wait(lck, [&] { return !ra_fetch_in_progress; });
	}
	bool use_large_buffers = !file_handle->PlainFileSource() && mode == ParserMode::PARSING;
	idx_t buffer_read_size = use_large_buffers ? INITIAL_BUFFER_SIZE_MAX : INITIAL_BUFFER_SIZE;
	idx_t maximum_line_size = use_large_buffers ? 2 * INITIAL_BUFFER_SIZE_MAX : options.maximum_line_size;

	if (read_ahead_buffer) {
		read_count = ra_buffer_size;
		buffer = std::move(read_ahead_buffer);
		buffer_size = read_count + options.maximum_line_size;

		if (remaining > options.maximum_line_size) {
			throw InvalidInputException("Maximum line size of %llu bytes exceeded!", options.maximum_line_size);
		}

		if (remaining > 0) {
			// remaining from last buffer: copy it before the new buffer
			memcpy(buffer.get() + options.maximum_line_size - remaining, old_buffer.get() + start, remaining);
		}

		start = options.maximum_line_size - remaining;
		position = options.maximum_line_size;

		read_ahead_buffer = nullptr;
		ra_buffer_size = 0;
	} else if (ra_is_eof) {
		if (remaining > 0) {
			throw InvalidInputException("Unexpected EOF");
		}
		buffer_size = 0;
		read_count = 0;
		start = 0;
		position = 0;
		if (old_buffer) {
			cached_buffers.push_back(move(old_buffer));
		}
		return false;
	} else {
		while (remaining > buffer_read_size) {
			buffer_read_size *= 2;
		}
		if (remaining + buffer_read_size > maximum_line_size) {
			// Note that while use_large_buffers == true, we actually allow larger line sizes. This should not matter to
			// the user though.
			throw InvalidInputException("Maximum line size of %llu bytes exceeded!", options.maximum_line_size);
		}
		buffer = unique_ptr<char[]>(new char[buffer_read_size + remaining + 1]);
		if (remaining > 0) {
			// remaining from last buffer: copy it here
			memcpy(buffer.get(), old_buffer.get() + start, remaining);
		}

		read_count = file_handle->Read(buffer.get() + remaining, buffer_read_size);
		start = 0;
		position = remaining;
		buffer_size = remaining + read_count;
	}

	// During parsing, we prefetch the next Buffer Asynchronously
	if (use_large_buffers && !ra_is_eof && read_count == buffer_read_size) {
		// Note that we reserve an extra options.maximum_line_size here to prevent an extra copy later as this
		// guarantees we can copy the remaining bytes into this buffer
		size_t read_ahead_buffer_alloc_size = buffer_read_size + options.maximum_line_size + 1;

		ra_fetch_in_progress = true;
		thread t(
		    [&](size_t read_size, size_t alloc_size) {
			    read_ahead_buffer = unique_ptr<char[]>(new char[alloc_size]);
			    ra_buffer_size = file_handle->Read(read_ahead_buffer.get() + options.maximum_line_size, read_size);
			    ra_is_eof = ra_buffer_size != read_size;
			    ra_fetch_in_progress = false;
			    ra_fetch_cv.notify_one();
		    },
		    buffer_read_size, read_ahead_buffer_alloc_size);

		t.detach();
	}

	bytes_in_chunk += read_count;
	buffer[buffer_size] = '\0';
	if (old_buffer) {
		cached_buffers.push_back(move(old_buffer));
	}
	if (!bom_checked) {
		bom_checked = true;
		if (read_count >= 3 && buffer[0] == '\xEF' && buffer[1] == '\xBB' && buffer[2] == '\xBF') {
			position += 3;
		}
	}

	return read_count > 0;
}
