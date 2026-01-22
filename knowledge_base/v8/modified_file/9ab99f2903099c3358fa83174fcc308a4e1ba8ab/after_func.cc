void Heap::ConfigureHeap(const v8::ResourceConstraints& constraints) {
  // Initialize max_semi_space_size_.
  {
    max_semi_space_size_ = 8 * (kSystemPointerSize / 4) * MB;
    if (constraints.max_young_generation_size_in_bytes() > 0) {
      max_semi_space_size_ = SemiSpaceSizeFromYoungGenerationSize(
          constraints.max_young_generation_size_in_bytes());
    }
    if (FLAG_max_semi_space_size > 0) {
      max_semi_space_size_ = static_cast<size_t>(FLAG_max_semi_space_size) * MB;
    } else if (FLAG_max_heap_size > 0) {
      size_t max_heap_size = static_cast<size_t>(FLAG_max_heap_size) * MB;
      size_t young_generation_size, old_generation_size;
      if (FLAG_max_old_space_size > 0) {
        old_generation_size = static_cast<size_t>(FLAG_max_old_space_size) * MB;
        young_generation_size = max_heap_size > old_generation_size
                                    ? max_heap_size - old_generation_size
                                    : 0;
      } else {
        GenerationSizesFromHeapSize(max_heap_size, &young_generation_size,
                                    &old_generation_size);
      }
      max_semi_space_size_ =
          SemiSpaceSizeFromYoungGenerationSize(young_generation_size);
    }
    if (FLAG_stress_compaction) {
      // This will cause more frequent GCs when stressing.
      max_semi_space_size_ = MB;
    }
    // TODO(dinfuehr): Rounding to a power of 2 is not longer needed. Remove it.
    max_semi_space_size_ =
        static_cast<size_t>(base::bits::RoundUpToPowerOfTwo64(
            static_cast<uint64_t>(max_semi_space_size_)));
    max_semi_space_size_ = Max(max_semi_space_size_, kMinSemiSpaceSize);
    max_semi_space_size_ = RoundDown<Page::kPageSize>(max_semi_space_size_);
  }

  // Initialize max_old_generation_size_ and max_global_memory_.
  {
    max_old_generation_size_ = 700ul * (kSystemPointerSize / 4) * MB;
    if (constraints.max_old_generation_size_in_bytes() > 0) {
      max_old_generation_size_ = constraints.max_old_generation_size_in_bytes();
    }
    if (FLAG_max_old_space_size > 0) {
      max_old_generation_size_ =
          static_cast<size_t>(FLAG_max_old_space_size) * MB;
    } else if (FLAG_max_heap_size > 0) {
      size_t max_heap_size = static_cast<size_t>(FLAG_max_heap_size) * MB;
      size_t young_generation_size =
          YoungGenerationSizeFromSemiSpaceSize(max_semi_space_size_);
      max_old_generation_size_ = max_heap_size > young_generation_size
                                     ? max_heap_size - young_generation_size
                                     : 0;
    }
    max_old_generation_size_ =
        Max(max_old_generation_size_, MinOldGenerationSize());
    max_old_generation_size_ =
        RoundDown<Page::kPageSize>(max_old_generation_size_);

    max_global_memory_size_ =
        GlobalMemorySizeFromV8Size(max_old_generation_size_);
  }

  CHECK_IMPLIES(FLAG_max_heap_size > 0,
                FLAG_max_semi_space_size == 0 || FLAG_max_old_space_size == 0);

  // Initialize initial_semispace_size_.
  {
    initial_semispace_size_ = kMinSemiSpaceSize;
    if (max_semi_space_size_ == kMaxSemiSpaceSize) {
      // Start with at least 1*MB semi-space on machines with a lot of memory.
      initial_semispace_size_ =
          Max(initial_semispace_size_, static_cast<size_t>(1 * MB));
    }
    if (constraints.initial_young_generation_size_in_bytes() > 0) {
      initial_semispace_size_ = SemiSpaceSizeFromYoungGenerationSize(
          constraints.initial_young_generation_size_in_bytes());
    }
    if (FLAG_initial_heap_size > 0) {
      size_t young_generation, old_generation;
      Heap::GenerationSizesFromHeapSize(
          static_cast<size_t>(FLAG_initial_heap_size) * MB, &young_generation,
          &old_generation);
      initial_semispace_size_ =
          SemiSpaceSizeFromYoungGenerationSize(young_generation);
    }
    if (FLAG_min_semi_space_size > 0) {
      initial_semispace_size_ =
          static_cast<size_t>(FLAG_min_semi_space_size) * MB;
    }
    initial_semispace_size_ =
        Min(initial_semispace_size_, max_semi_space_size_);
    initial_semispace_size_ =
        RoundDown<Page::kPageSize>(initial_semispace_size_);
  }

  // Initialize initial_old_space_size_.
  {
    initial_old_generation_size_ = kMaxInitialOldGenerationSize;
    if (constraints.initial_old_generation_size_in_bytes() > 0) {
      initial_old_generation_size_ =
          constraints.initial_old_generation_size_in_bytes();
      old_generation_size_configured_ = true;
    }
    if (FLAG_initial_heap_size > 0) {
      size_t initial_heap_size =
          static_cast<size_t>(FLAG_initial_heap_size) * MB;
      size_t young_generation_size =
          YoungGenerationSizeFromSemiSpaceSize(initial_semispace_size_);
      initial_old_generation_size_ =
          initial_heap_size > young_generation_size
              ? initial_heap_size - young_generation_size
              : 0;
      old_generation_size_configured_ = true;
    }
    if (FLAG_initial_old_space_size > 0) {
      initial_old_generation_size_ =
          static_cast<size_t>(FLAG_initial_old_space_size) * MB;
      old_generation_size_configured_ = true;
    }
    initial_old_generation_size_ =
        Min(initial_old_generation_size_, max_old_generation_size_ / 2);
    initial_old_generation_size_ =
        RoundDown<Page::kPageSize>(initial_old_generation_size_);
  }

  if (old_generation_size_configured_) {
    // If the embedder pre-configures the initial old generation size,
    // then allow V8 to skip full GCs below that threshold.
    min_old_generation_size_ = initial_old_generation_size_;
    min_global_memory_size_ =
        GlobalMemorySizeFromV8Size(min_old_generation_size_);
  }

  if (FLAG_semi_space_growth_factor < 2) {
    FLAG_semi_space_growth_factor = 2;
  }

  old_generation_allocation_limit_ = initial_old_generation_size_;
  global_allocation_limit_ =
      GlobalMemorySizeFromV8Size(old_generation_allocation_limit_);
  initial_max_old_generation_size_ = max_old_generation_size_;

  // We rely on being able to allocate new arrays in paged spaces.
  DCHECK(kMaxRegularHeapObjectSize >=
         (JSArray::kHeaderSize +
          FixedArray::SizeFor(JSArray::kInitialMaxFastElementArray) +
          AllocationMemento::kSize));

  code_range_size_ = constraints.code_range_size_in_bytes();

  configured_ = true;
}
