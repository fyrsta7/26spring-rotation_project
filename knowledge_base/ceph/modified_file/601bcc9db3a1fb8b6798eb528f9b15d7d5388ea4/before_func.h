
  LogSegment *peek_current_segment() {
    return segments.empty() ? NULL : segments.rbegin()->second;
  }

