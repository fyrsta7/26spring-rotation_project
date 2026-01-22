void Utf8ExternalStreamingStream::FillBufferFromCurrentChunk() {
  DCHECK_LT(current_.chunk_no, chunks_.size());
  DCHECK_EQ(buffer_start_, buffer_cursor_);
  DCHECK_LT(buffer_end_ + 1, buffer_start_ + kBufferSize);

  const Chunk& chunk = chunks_[current_.chunk_no];

  // The buffer_ is writable, but buffer_*_ members are const. So we get a
  // non-const pointer into buffer that points to the same char as buffer_end_.
  uint16_t* output_cursor = buffer_ + (buffer_end_ - buffer_start_);
  DCHECK_EQ(output_cursor, buffer_end_);

  unibrow::Utf8::State state = current_.pos.state;
  uint32_t incomplete_char = current_.pos.incomplete_char;

  // If the current chunk is the last (empty) chunk we'll have to process
  // any left-over, partial characters.
  if (chunk.length == 0) {
    unibrow::uchar t = unibrow::Utf8::ValueOfIncrementalFinish(&state);
    if (t != unibrow::Utf8::kBufferEmpty) {
      DCHECK_EQ(t, unibrow::Utf8::kBadChar);
      *output_cursor = static_cast<uc16>(t);
      buffer_end_++;
      current_.pos.chars++;
      current_.pos.incomplete_char = 0;
      current_.pos.state = state;
    }
    return;
  }

  size_t it = current_.pos.bytes - chunk.start.bytes;
  const uint8_t* cursor = chunk.data + it;
  const uint8_t* end = chunk.data + chunk.length;

  // Deal with possible BOM.
  if (V8_UNLIKELY(current_.pos.bytes < 3 && current_.pos.chars == 0)) {
    while (cursor < end) {
      unibrow::uchar t =
          unibrow::Utf8::ValueOfIncremental(&cursor, &state, &incomplete_char);
      if (V8_LIKELY(t < kUtf8Bom)) {
        *(output_cursor++) = static_cast<uc16>(t);  // The most frequent case.
      } else if (t == unibrow::Utf8::kIncomplete) {
        continue;
      } else if (t == kUtf8Bom) {
        // BOM detected at beginning of the stream. Don't copy it.
      } else if (t <= unibrow::Utf16::kMaxNonSurrogateCharCode) {
        *(output_cursor++) = static_cast<uc16>(t);
      } else {
        *(output_cursor++) = unibrow::Utf16::LeadSurrogate(t);
        *(output_cursor++) = unibrow::Utf16::TrailSurrogate(t);
      }
      break;
    }
  }

  const uint16_t* max_buffer_end = buffer_start_ + kBufferSize;
  while (cursor < end && output_cursor + 1 < max_buffer_end) {
    unibrow::uchar t =
        unibrow::Utf8::ValueOfIncremental(&cursor, &state, &incomplete_char);
    if (V8_LIKELY(t <= unibrow::Utf16::kMaxNonSurrogateCharCode)) {
      *(output_cursor++) = static_cast<uc16>(t);  // The most frequent case.
    } else if (t == unibrow::Utf8::kIncomplete) {
      continue;
    } else {
      *(output_cursor++) = unibrow::Utf16::LeadSurrogate(t);
      *(output_cursor++) = unibrow::Utf16::TrailSurrogate(t);
    }
    // Fast path for ascii sequences.
    size_t remaining = end - cursor;
    size_t max_buffer = max_buffer_end - output_cursor;
    int max_length = static_cast<int>(Min(remaining, max_buffer));
    DCHECK_EQ(state, unibrow::Utf8::State::kAccept);
    const uint8_t* read_end = cursor + max_length;
    for (; cursor < read_end; cursor++) {
      uint8_t c = *cursor;
      DCHECK_EQ(unibrow::Utf8::kMaxOneByteChar, 0x7F);
      if (c > unibrow::Utf8::kMaxOneByteChar) break;
      *(output_cursor++) = c;
    }
  }

  current_.pos.bytes = chunk.start.bytes + (cursor - chunk.data);
  current_.pos.chars += (output_cursor - buffer_end_);
  current_.pos.incomplete_char = incomplete_char;
  current_.pos.state = state;
  current_.chunk_no += (cursor == end);

  buffer_end_ = output_cursor;
}
