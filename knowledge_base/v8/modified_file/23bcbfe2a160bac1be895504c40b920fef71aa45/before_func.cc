uint32_t V8HeapExplorer::EstimateObjectsCount() {
  CombinedHeapObjectIterator it(heap_, HeapObjectIterator::kFilterUnreachable);
  uint32_t objects_count = 0;
  // Avoid overflowing the objects count. In worst case, we will show the same
  // progress for a longer period of time, but we do not expect to have that
  // many objects.
  while (!it.Next().is_null() &&
         objects_count != std::numeric_limits<uint32_t>::max())
    ++objects_count;
  return objects_count;
}
