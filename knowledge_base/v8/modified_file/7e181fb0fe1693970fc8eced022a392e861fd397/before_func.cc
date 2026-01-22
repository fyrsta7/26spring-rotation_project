void SamplingHeapProfiler::SampleObject(Address soon_object, size_t size) {
  DisallowHeapAllocation no_allocation;

  HandleScope scope(isolate_);
  HeapObject heap_object = HeapObject::FromAddress(soon_object);
  Handle<Object> obj(heap_object, isolate_);

  // Mark the new block as FreeSpace to make sure the heap is iterable while we
  // are taking the sample.
  heap_->CreateFillerObjectAt(soon_object, static_cast<int>(size),
                              ClearRecordedSlots::kNo);

  Local<v8::Value> loc = v8::Utils::ToLocal(obj);

  AllocationNode* node = AddStack();
  node->allocations_[size]++;
  auto sample =
      std::make_unique<Sample>(size, node, loc, this, next_sample_id());
  sample->global.SetWeak(sample.get(), OnWeakCallback,
                         WeakCallbackType::kParameter);
  samples_.emplace(sample.get(), std::move(sample));
}
