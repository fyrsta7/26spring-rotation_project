  static void MoveElements(Isolate* isolate, Handle<JSArray> receiver,
                           Handle<FixedArrayBase> backing_store, int dst_index,
                           int src_index, int len, int hole_start,
                           int hole_end) {
    Heap* heap = isolate->heap();
    Handle<BackingStore> dst_elms = Handle<BackingStore>::cast(backing_store);
    if (len > JSArray::kMaxCopyElements && dst_index == 0 &&
        heap->CanMoveObjectStart(*dst_elms)) {
      // Update all the copies of this backing_store handle.
      *dst_elms.location() =
          BackingStore::cast(heap->LeftTrimFixedArray(*dst_elms, src_index));
      receiver->set_elements(*dst_elms);
      // Adjust the hole offset as the array has been shrunk.
      hole_end -= src_index;
      DCHECK_LE(hole_start, backing_store->length());
      DCHECK_LE(hole_end, backing_store->length());
    } else if (len != 0) {
      if (IsDoubleElementsKind(KindTraits::Kind)) {
        MemMove(dst_elms->data_start() + dst_index,
                dst_elms->data_start() + src_index, len * kDoubleSize);
      } else {
        DisallowHeapAllocation no_gc;
        heap->MoveElements(FixedArray::cast(*dst_elms), dst_index, src_index,
                           len);
      }
    }
    if (hole_start != hole_end) {
      dst_elms->FillWithHoles(hole_start, hole_end);
    }
  }
