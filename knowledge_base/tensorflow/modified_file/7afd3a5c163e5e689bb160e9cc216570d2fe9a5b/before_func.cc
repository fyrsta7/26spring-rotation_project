AlternateMemoryBestFitHeap::GetLiveAllocationAt(
    const MemorySpaceAssignment::AllocationSequence& allocations,
    int64_t time) {
  for (auto allocation_it = allocations.rbegin();
       allocation_it != allocations.rend(); ++allocation_it) {
    if ((*allocation_it)->start_time() <= time &&
        (*allocation_it)->end_time() >= time) {
      return allocation_it->get();
    }
  }
  return nullptr;
}

void AlternateMemoryBestFitHeap::AllocateCrossProgramPrefetchBuffer(
    HloModule* module, const BufferInterval& prefetch_candidate) {
  Chunk chunk_candidate = FindChunkCandidate(prefetch_candidate);
  if (chunk_candidate.chunk_end() > available_heap_size()) {
    VLOG(3) << "Could not allocate preferred memory for cross program prefetch";
    return;
  }

  const HloValue* buffer = prefetch_candidate.buffer;
  int64_t parameter = buffer->instruction()->parameter_number();
  int cross_program_prefetch_index = module->CrossProgramPrefetches().size();
  module->AddCrossProgramPrefetch(parameter, buffer->index());

  MemorySpaceAssignment::AllocationSequence allocations;
  allocations.push_back(std::make_unique<MemorySpaceAssignment::Allocation>(
      buffer->defining_position(), MemorySpace::kDefault, kDummyChunk,
      prefetch_candidate.start, prefetch_candidate.end,
      /*is_scoped_allocation=*/false));

  // Find the earliest use.
  const auto& instruction_schedule = hlo_live_range_.instruction_schedule();
  auto uses = FindCrossProgramPrefetchUses(buffer->GetUses());
  CHECK_GE(uses.size(), 1);
  auto use_schedule_compare = [&](const HloUse& lhs, const HloUse& rhs) {
    return instruction_schedule.at(lhs.instruction) <
           instruction_schedule.at(rhs.instruction);
  };
  auto first_use = absl::c_min_element(uses, use_schedule_compare);
  int64_t latest_prefetch_time =
      instruction_schedule.at(first_use->instruction);

  // Find the latest use time.
  int64_t last_use_time = instruction_schedule.at(
      absl::c_max_element(uses, use_schedule_compare)->instruction);
  for (const HloValue* colocation : prefetch_candidate.colocations) {
    auto colocation_uses = colocation->GetUses();
    if (!colocation_uses.empty()) {
      last_use_time = std::max(
          last_use_time,
          instruction_schedule.at(
              absl::c_max_element(colocation_uses, use_schedule_compare)
                  ->instruction));
    }
  }

  int64_t end_of_program_prefetch_end_time = instruction_schedule.size();
  int64_t end_of_program_prefetch_latest_start_time =
      options_.prefetch_interval_picker->LatestPrefetchStartTime(
          buffer->defining_position().shape(), last_use_time,
          end_of_program_prefetch_end_time, nullptr);
  int64_t end_of_program_prefetch_start_time =
      options_.prefetch_interval_picker->PreferredPrefetchStartTime(
          buffer->defining_position().shape(), last_use_time,
          end_of_program_prefetch_latest_start_time,
          end_of_program_prefetch_end_time);
  VLOG(2) << "last use time = " << last_use_time
          << ", end-of-program prefetch start time = "
          << end_of_program_prefetch_start_time;
  float total_execution_time =
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          0, instruction_schedule.size());
  float buffer_occupied_time =
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          0, last_use_time) +
      options_.prefetch_interval_picker->GetLogicalIntervalElapsed(
          end_of_program_prefetch_start_time, end_of_program_prefetch_end_time);
  float buffer_occupied_ratio = buffer_occupied_time / total_execution_time;
  VLOG(2) << "Total execution time = " << total_execution_time
          << ", buffer occupied time = " << buffer_occupied_time
          << ", buffer occupied ratio = " << buffer_occupied_ratio;
  // Freeing buffer only makes sense if the buffer will be free for a
  // substantial time. Only perform this optimization if the ratio is below the
  // limit, and if the memory pressure is above the alternate memory size.
  bool free_buffer =
      (options_.enable_cross_program_prefetch_freeing &&
       memory_pressure_ > options_.max_size_in_bytes &&
       buffer_occupied_ratio < kCrossProgramPrefetchOccupyFreeingLimit &&
       end_of_program_prefetch_start_time > last_use_time &&
       end_of_program_prefetch_start_time < end_of_program_prefetch_end_time);
  int64_t cross_program_prefetch_end_time =
      free_buffer ? last_use_time : prefetch_candidate.end;

  AddAsyncCopy(*allocations.back(), MemorySpace::kAlternate, chunk_candidate,
               prefetch_candidate.start, cross_program_prefetch_end_time,
               latest_prefetch_time, &allocations, /*aliased_offset=*/nullptr,
               /*resource=*/0.0, cross_program_prefetch_index);

  absl::c_for_each(uses, [&](auto& use) { allocations.back()->AddUse(use); });
  AliasedOffset* cross_program_prefetch_offset =
      GetAliasedOffset(*allocations.back());

  if (free_buffer) {
    VLOG(2) << "Adding an end-of-program prefetch for freed "
               "cross-program-prefetched buffer.";
    AddAsyncCopy(*allocations.front(), MemorySpace::kAlternate, chunk_candidate,
                 end_of_program_prefetch_start_time,
                 end_of_program_prefetch_end_time,
                 end_of_program_prefetch_end_time, &allocations,
                 cross_program_prefetch_offset,
                 /*resource=*/0.0);
    CHECK_EQ(cross_program_prefetch_offset->offset,
             allocations.back()->chunk().offset);
  }

  const int allocations_initial_size = allocations_->size();
  for (auto& allocation : allocations) {
    if (allocation->memory_space() == MemorySpace::kAlternate) {
      BufferInterval buffer_interval;
      buffer_interval.start = allocation->start_time();
      buffer_interval.end = allocation->end_time();
      buffer_interval.size = allocation->chunk().size;
      buffer_interval.buffer = prefetch_candidate.buffer;
      AddToPendingChunks(buffer_interval, chunk_candidate);
    }
    allocations_->push_back(std::move(allocation));
  }

  // Add a repack allocation block for the Allocation objects in alternate
  // memory.
  std::vector<RepackAllocationBlock*> colocations;
  for (int i = allocations_initial_size; i < allocations_->size(); ++i) {
    const auto& allocation = allocations_->at(i);
    if (allocation->memory_space() == MemorySpace::kAlternate) {
      repack_allocation_blocks_.push_back(MakeRepackAllocationBlock(
          allocation->start_time(), allocation->end_time(),
          allocation->chunk().size, allocation->chunk().offset,
