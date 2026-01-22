
  LogSegment *peek_current_segment() {
    return segments.empty() ? NULL : segments.rbegin()->second;
  }

  LogSegment *get_current_segment() { 
    ceph_assert(!segments.empty());
    return segments.rbegin()->second;
