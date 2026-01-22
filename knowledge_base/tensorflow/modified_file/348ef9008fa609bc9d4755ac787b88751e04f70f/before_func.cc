    for (int i = 0; i < callable_options.fetch().size(); ++i) {
      const string& output = callable_options.fetch(i);
      ek->output_name_to_rendezvous_key[output] =
          GetRendezvousKey(output, device_set_.client_device()->attributes(),
                           FrameAndIter(0, 0));
    }
  }

  *out_executors_and_keys = std::move(ek);
  *out_func_info = std::move(func_info);
  return Status::OK();
}

Status DirectSession::GetOrCreateExecutors(
    gtl::ArraySlice<string> inputs, gtl::ArraySlice<string> outputs,
    gtl::ArraySlice<string> target_nodes, ExecutorsAndKeys** executors_and_keys,
    RunStateArgs* run_state_args) {
  int64 handle_name_counter_value = -1;
  if (LogMemory::IsEnabled() || run_state_args->is_partial_run) {
    handle_name_counter_value = handle_name_counter_.fetch_add(1);
  }

  string debug_tensor_watches_summary;
  if (!run_state_args->debug_options.debug_tensor_watch_opts().empty()) {
    debug_tensor_watches_summary = SummarizeDebugTensorWatches(
        run_state_args->debug_options.debug_tensor_watch_opts());
  }

  // Fast lookup path, no sorting.
  const string key = strings::StrCat(
      absl::StrJoin(inputs, ","), "->", absl::StrJoin(outputs, ","), "/",
      absl::StrJoin(target_nodes, ","), "/", run_state_args->is_partial_run,
      "/", debug_tensor_watches_summary);
  // Set the handle, if it's needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(key, ";", handle_name_counter_value);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);  // could use reader lock
    auto it = executors_.find(key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      return Status::OK();
    }
  }

  // Slow lookup path, the unsorted key missed the cache.
  // Sort the inputs and outputs, and look up with the sorted key in case an
  // earlier call used a different order of inputs and outputs.
  //
  // We could consider some other signature instead of sorting that
  // preserves the same property to avoid the sort in the future.
  std::vector<string> inputs_sorted(inputs.begin(), inputs.end());
  std::sort(inputs_sorted.begin(), inputs_sorted.end());
  std::vector<string> outputs_sorted(outputs.begin(), outputs.end());
  std::sort(outputs_sorted.begin(), outputs_sorted.end());
  std::vector<string> tn_sorted(target_nodes.begin(), target_nodes.end());
  std::sort(tn_sorted.begin(), tn_sorted.end());

  const string sorted_key = strings::StrCat(
      absl::StrJoin(inputs_sorted, ","), "->",
      absl::StrJoin(outputs_sorted, ","), "/", absl::StrJoin(tn_sorted, ","),
      "/", run_state_args->is_partial_run, "/", debug_tensor_watches_summary);
  // Set the handle, if its needed to log memory or for partial run.
  if (handle_name_counter_value >= 0) {
    run_state_args->handle =
        strings::StrCat(sorted_key, ";", handle_name_counter_value);
  }

  // See if we already have the executors for this run.
  {
    mutex_lock l(executor_lock_);
    auto it = executors_.find(sorted_key);
    if (it != executors_.end()) {
      *executors_and_keys = it->second.get();
      // Insert this under the original key.
      executors_.emplace(key, it->second);
      return Status::OK();
    }
  }

  // Nothing found, so create the executors and store in the cache.
  // The executor_lock_ is intentionally released while executors are
  // being created.
  CallableOptions callable_options;
  callable_options.mutable_feed()->Reserve(inputs_sorted.size());
  for (const string& input : inputs_sorted) {
    callable_options.add_feed(input);
  }
  callable_options.mutable_fetch()->Reserve(outputs_sorted.size());
  for (const string& output : outputs_sorted) {
    callable_options.add_fetch(output);
  }
  callable_options.mutable_target()->Reserve(tn_sorted.size());
  for (const string& target : tn_sorted) {
    callable_options.add_target(target);
  }
  *callable_options.mutable_run_options()->mutable_debug_options() =
      run_state_args->debug_options;
  callable_options.mutable_run_options()
      ->mutable_experimental()
      ->set_collective_graph_key(run_state_args->collective_graph_key);
  std::unique_ptr<ExecutorsAndKeys> ek;
  std::unique_ptr<FunctionInfo> func_info;
  TF_RETURN_IF_ERROR(
      CreateExecutors(callable_options, &ek, &func_info, run_state_args));

  // Reacquire the lock, try to insert into the map.
  mutex_lock l(executor_lock_);

  // Another thread may have created the entry before us, in which case we will
  // reuse the already created one.
