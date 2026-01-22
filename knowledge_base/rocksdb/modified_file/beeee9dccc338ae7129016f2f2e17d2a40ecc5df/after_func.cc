Status DBImpl::DoCompactionWork(CompactionState* compact,
                                DeletionState& deletion_state,
                                LogBuffer* log_buffer) {
  assert(compact);
  compact->CleanupBatchBuffer();
  compact->CleanupMergedBuffer();
  bool prefix_initialized = false;

  int64_t imm_micros = 0;  // Micros spent doing imm_ compactions
  ColumnFamilyData* cfd = compact->compaction->column_family_data();
  LogToBuffer(
      log_buffer,
      "[CF %u] Compacting %d@%d + %d@%d files, score %.2f slots available %d",
      cfd->GetID(), compact->compaction->num_input_files(0),
      compact->compaction->level(), compact->compaction->num_input_files(1),
      compact->compaction->output_level(), compact->compaction->score(),
      options_.max_background_compactions - bg_compaction_scheduled_);
  char scratch[2345];
  compact->compaction->Summary(scratch, sizeof(scratch));
  LogToBuffer(log_buffer, "Compaction start summary: %s\n", scratch);

  assert(cfd->current()->NumLevelFiles(compact->compaction->level()) > 0);
  assert(compact->builder == nullptr);
  assert(!compact->outfile);

  SequenceNumber visible_at_tip = 0;
  SequenceNumber earliest_snapshot;
  SequenceNumber latest_snapshot = 0;
  snapshots_.getAll(compact->existing_snapshots);
  if (compact->existing_snapshots.size() == 0) {
    // optimize for fast path if there are no snapshots
    visible_at_tip = versions_->LastSequence();
    earliest_snapshot = visible_at_tip;
  } else {
    latest_snapshot = compact->existing_snapshots.back();
    // Add the current seqno as the 'latest' virtual
    // snapshot to the end of this list.
    compact->existing_snapshots.push_back(versions_->LastSequence());
    earliest_snapshot = compact->existing_snapshots[0];
  }

  // Is this compaction producing files at the bottommost level?
  bool bottommost_level = compact->compaction->BottomMostLevel();

  // Allocate the output file numbers before we release the lock
  AllocateCompactionOutputFileNumbers(compact);

  // Release mutex while we're actually doing the compaction work
  mutex_.Unlock();
  log_buffer->FlushBufferToLog();

  const uint64_t start_micros = env_->NowMicros();
  unique_ptr<Iterator> input(versions_->MakeInputIterator(compact->compaction));
  input->SeekToFirst();
  shared_ptr<Iterator> backup_input(
      versions_->MakeInputIterator(compact->compaction));
  backup_input->SeekToFirst();

  Status status;
  ParsedInternalKey ikey;
  std::unique_ptr<CompactionFilterV2> compaction_filter_from_factory_v2
    = nullptr;
  auto context = compact->GetFilterContext();
  compaction_filter_from_factory_v2 =
      cfd->options()->compaction_filter_factory_v2->CreateCompactionFilterV2(
          context);
  auto compaction_filter_v2 =
    compaction_filter_from_factory_v2.get();

  // temp_backup_input always point to the start of the current buffer
  // temp_backup_input = backup_input;
  // iterate through input,
  // 1) buffer ineligible keys and value keys into 2 separate buffers;
  // 2) send value_buffer to compaction filter and alternate the values;
  // 3) merge value_buffer with ineligible_value_buffer;
  // 4) run the modified "compaction" using the old for loop.
  if (compaction_filter_v2) {
    while (backup_input->Valid() && !shutting_down_.Acquire_Load() &&
           !cfd->IsDropped()) {
      // FLUSH preempts compaction
      // TODO(icanadi) this currently only checks if flush is necessary on
      // compacting column family. we should also check if flush is necessary on
      // other column families, too
      imm_micros += CallFlushDuringCompaction(cfd, deletion_state, log_buffer);

      Slice key = backup_input->key();
      Slice value = backup_input->value();

      const SliceTransform* transformer =
          cfd->options()->compaction_filter_factory_v2->GetPrefixExtractor();
      const auto key_prefix = transformer->Transform(key);
      if (!prefix_initialized) {
        compact->cur_prefix_ = key_prefix.ToString();
        prefix_initialized = true;
      }
      if (!ParseInternalKey(key, &ikey)) {
        // log error
        Log(options_.info_log, "Failed to parse key: %s",
            key.ToString().c_str());
        continue;
      } else {
        // If the prefix remains the same, keep buffering
        if (key_prefix.compare(Slice(compact->cur_prefix_)) == 0) {
          // Apply the compaction filter V2 to all the kv pairs sharing
          // the same prefix
          if (ikey.type == kTypeValue &&
              (visible_at_tip || ikey.sequence > latest_snapshot)) {
            // Buffer all keys sharing the same prefix for CompactionFilterV2
            // Iterate through keys to check prefix
            compact->BufferKeyValueSlices(key, value);
          } else {
            // buffer ineligible keys
            compact->BufferOtherKeyValueSlices(key, value);
          }
          backup_input->Next();
          continue;
          // finish changing values for eligible keys
        } else {
          // Now prefix changes, this batch is done.
          // Call compaction filter on the buffered values to change the value
          if (compact->key_buf_.size() > 0) {
            CallCompactionFilterV2(compact, compaction_filter_v2);
          }
          compact->cur_prefix_ = key_prefix.ToString();
        }
      }

      // Merge this batch of data (values + ineligible keys)
      compact->MergeKeyValueSliceBuffer(&cfd->internal_comparator());

      // Done buffering for the current prefix. Spit it out to disk
      // Now just iterate through all the kv-pairs
      status = ProcessKeyValueCompaction(
          visible_at_tip,
          earliest_snapshot,
          latest_snapshot,
          deletion_state,
          bottommost_level,
          imm_micros,
          input.get(),
          compact,
          true,
          log_buffer);

      if (!status.ok()) {
        break;
      }

      // After writing the kv-pairs, we can safely remove the reference
      // to the string buffer and clean them up
      compact->CleanupBatchBuffer();
      compact->CleanupMergedBuffer();
      // Buffer the key that triggers the mismatch in prefix
      if (ikey.type == kTypeValue &&
        (visible_at_tip || ikey.sequence > latest_snapshot)) {
        compact->BufferKeyValueSlices(key, value);
      } else {
        compact->BufferOtherKeyValueSlices(key, value);
      }
      backup_input->Next();
      if (!backup_input->Valid()) {
        // If this is the single last value, we need to merge it.
        if (compact->key_buf_.size() > 0) {
          CallCompactionFilterV2(compact, compaction_filter_v2);
        }
        compact->MergeKeyValueSliceBuffer(&cfd->internal_comparator());

        status = ProcessKeyValueCompaction(
            visible_at_tip,
            earliest_snapshot,
            latest_snapshot,
            deletion_state,
            bottommost_level,
            imm_micros,
            input.get(),
            compact,
            true,
            log_buffer);

        compact->CleanupBatchBuffer();
        compact->CleanupMergedBuffer();
      }
    }  // done processing all prefix batches
    // finish the last batch
    if (compact->key_buf_.size() > 0) {
      CallCompactionFilterV2(compact, compaction_filter_v2);
    }
    compact->MergeKeyValueSliceBuffer(&cfd->internal_comparator());
    status = ProcessKeyValueCompaction(
        visible_at_tip,
        earliest_snapshot,
        latest_snapshot,
        deletion_state,
        bottommost_level,
        imm_micros,
        input.get(),
        compact,
        true,
        log_buffer);
  }  // checking for compaction filter v2

  if (!compaction_filter_v2) {
    status = ProcessKeyValueCompaction(
      visible_at_tip,
      earliest_snapshot,
      latest_snapshot,
      deletion_state,
      bottommost_level,
      imm_micros,
      input.get(),
      compact,
      false,
      log_buffer);
  }

  if (status.ok() && (shutting_down_.Acquire_Load() || cfd->IsDropped())) {
    status = Status::ShutdownInProgress(
        "Database shutdown or Column family drop during compaction");
  }
  if (status.ok() && compact->builder != nullptr) {
    status = FinishCompactionOutputFile(compact, input.get());
  }
  if (status.ok()) {
    status = input->status();
  }
  input.reset();

  if (!options_.disableDataSync) {
    db_directory_->Fsync();
  }

  InternalStats::CompactionStats stats;
  stats.micros = env_->NowMicros() - start_micros - imm_micros;
  MeasureTime(options_.statistics.get(), COMPACTION_TIME, stats.micros);
  stats.files_in_leveln = compact->compaction->num_input_files(0);
  stats.files_in_levelnp1 = compact->compaction->num_input_files(1);

  int num_output_files = compact->outputs.size();
  if (compact->builder != nullptr) {
    // An error occurred so ignore the last output.
    assert(num_output_files > 0);
    --num_output_files;
  }
  stats.files_out_levelnp1 = num_output_files;

  for (int i = 0; i < compact->compaction->num_input_files(0); i++) {
    stats.bytes_readn += compact->compaction->input(0, i)->file_size;
    RecordTick(options_.statistics.get(), COMPACT_READ_BYTES,
               compact->compaction->input(0, i)->file_size);
  }

  for (int i = 0; i < compact->compaction->num_input_files(1); i++) {
    stats.bytes_readnp1 += compact->compaction->input(1, i)->file_size;
    RecordTick(options_.statistics.get(), COMPACT_READ_BYTES,
               compact->compaction->input(1, i)->file_size);
  }

  for (int i = 0; i < num_output_files; i++) {
    stats.bytes_written += compact->outputs[i].file_size;
    RecordTick(options_.statistics.get(), COMPACT_WRITE_BYTES,
               compact->outputs[i].file_size);
  }

  LogFlush(options_.info_log);
  mutex_.Lock();
  cfd->internal_stats()->AddCompactionStats(compact->compaction->output_level(),
                                            stats);

  // if there were any unused file number (mostly in case of
  // compaction error), free up the entry from pending_putputs
  ReleaseCompactionUnusedFileNumbers(compact);

  if (status.ok()) {
    status = InstallCompactionResults(compact, log_buffer);
    InstallSuperVersion(cfd, deletion_state);
  }
  Version::LevelSummaryStorage tmp;
  LogToBuffer(
      log_buffer,
      "compacted to: %s, %.1f MB/sec, level %d, files in(%d, %d) out(%d) "
      "MB in(%.1f, %.1f) out(%.1f), read-write-amplify(%.1f) "
      "write-amplify(%.1f) %s\n",
      cfd->current()->LevelSummary(&tmp),
      (stats.bytes_readn + stats.bytes_readnp1 + stats.bytes_written) /
          (double)stats.micros,
      compact->compaction->output_level(), stats.files_in_leveln,
      stats.files_in_levelnp1, stats.files_out_levelnp1,
      stats.bytes_readn / 1048576.0, stats.bytes_readnp1 / 1048576.0,
      stats.bytes_written / 1048576.0,
      (stats.bytes_written + stats.bytes_readnp1 + stats.bytes_readn) /
          (double)stats.bytes_readn,
      stats.bytes_written / (double)stats.bytes_readn,
      status.ToString().c_str());

  return status;
}
