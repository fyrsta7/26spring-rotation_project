        py::reinterpret_borrow<py::object>(out_array_sharding));
  }

  py::list out_committed = pmap_data.attr("out_committed");

  DCHECK(out_committed.empty() || out_avals.size() == out_committed.size());

  cache_entry.out_committed.reserve(out_committed.size());
  for (py::handle c : out_committed) {
    cache_entry.out_committed.push_back(py::cast<bool>(c));
  }
}

xla::StatusOr<py::object> PmapFunction::Call(py::handle callable,
                                             PyObject* const* args,
                                             size_t nargs, PyObject* kwnames) {
  // Calls the cache_miss_ function. This just calls the Python function; it may
  // return nullptr value if a Python exception is thrown.
  auto cache_miss = [&]() -> py::tuple {
    return py::reinterpret_steal<py::tuple>(
        JAX_PyObject_Vectorcall(cache_miss_.ptr(), args, nargs, kwnames));
  };

  // Call the cache_miss() function, extracting the output data and ignoring
  // the fastpath data. If the cache miss returns a Python error, returns
  // nullptr and leaves the Python error set.
  auto fallback_to_cache_miss = [&]() {
    py::tuple cache_miss_output = cache_miss();
    if (!cache_miss_output.ptr()) {
      return py::object();
    }
    return py::object(cache_miss_output[0]);
  };

  if (always_fallback_to_python_) {
    return fallback_to_cache_miss();
  }

  size_t num_positional_args = PyVectorcall_NARGS(nargs);
  size_t num_keyword_args = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
  absl::Span<PyObject* const> positional_args(args, num_positional_args);
  absl::Span<PyObject* const> keyword_args(args + num_positional_args,
                                           num_keyword_args);
  ParsedArgumentsAsBuffers arguments;
  xla::Status status =
      ParseArguments(positional_args, keyword_args, kwnames, static_argnums_,
                     /*static_argnames=*/{}, arguments);
  if (!status.ok()) {
    VLOG(2) << "ParseArguments failed: " << status;
    return fallback_to_cache_miss();
  }

  status = UpdateArgsSignature(arguments);
  if (!status.ok()) {
    return fallback_to_cache_miss();
  }

  // Retrieve/Maybe add the executable to the cache.
  absl::flat_hash_map<CallSignature, std::unique_ptr<PmapCacheEntry>>::iterator
      it;
  bool inserted;
  std::tie(it, inserted) = executables_.try_emplace(
      arguments.signature, std::unique_ptr<PmapCacheEntry>());
  if (inserted) {
    it->second = std::make_unique<PmapCacheEntry>();
  }
  PmapCacheEntry& cache_entry = *(it->second);

  if (!cache_entry.compilation_complete.HasBeenNotified()) {
    // In case of several threads attempting to compile the executable, only
    // the one that inserted the item will perform the compilation.
    if (inserted) {
      py::object out_and_fastpath_data;
      py::tuple out_tuple;
      VLOG(2) << "Cache miss for " << arguments.signature.DebugString();
      try {
        // Calls Python and may release the GIL. May also throw if
        // compilation/tracing fails.
        out_and_fastpath_data = cache_miss();
        if (!out_and_fastpath_data.ptr()) {
          throw py::error_already_set();
        }
        out_tuple = py::cast<py::tuple>(out_and_fastpath_data);
        PopulateCacheEntry(cache_entry, arguments.signature, out_tuple);
      } catch (const std::exception& e) {
        cache_entry.fall_back_to_python = true;
        cache_entry.compilation_complete.Notify();
        throw;
      }
      cache_entry.compilation_complete.Notify();

      // We have already computed the result in the miss path so we can return
      // it. We are even *required* to do so if there are donated arguments,
      // because any donated buffers will now be invalid.
      return py::object(out_tuple[0]);
    } else {
      // Release the GIL while we wait, making sure the compile thread can
      // lock it.
      py::gil_scoped_release release;
      cache_entry.compilation_complete.WaitForNotification();
    }
  }
  if (cache_entry.fall_back_to_python) {
    return fallback_to_cache_miss();
  }

  // 1. Parse arguments.
  std::vector<xla::PjRtDevice*>& input_devices = cache_entry.devices;
  const int num_computations =
      cache_entry.executable->AddressableDevices().size();
  std::vector<InputSpec>& input_specs = cache_entry.input_specs;
  const int num_args = arguments.flat_dynamic_args.size();

#ifdef JAX_ENABLE_IFRT
  // We need [num_args] for the `Execute` call below.
  std::vector<tsl::RCReference<xla::ifrt::Array>> num_args_arrays(num_args);
  for (int i = 0; i < num_args; ++i) {
    TF_ASSIGN_OR_RETURN(
        ShardArgResult sharded_arg,
        ShardArg(arguments.flat_dynamic_args[i], input_devices, input_specs[i],
                 cache_entry.py_devices, python_shard_arg_fallback_));

    num_args_arrays[i] = std::move(sharded_arg.ifrt_array);
    if (sharded_arg.owning_sda) {
      arguments.keep_alive_objects.push_back(std::move(sharded_arg.owning_sda));
    }
  }
#else
  // We need [num_computation, num_args] for the `Execute` call bellow,
  std::vector<std::vector<xla::PjRtBuffer*>> num_computation_num_args_buffers(
      num_computations);
  for (int computation = 0; computation < num_computations; ++computation) {
    num_computation_num_args_buffers[computation].resize(num_args);
  }
  for (int i = 0; i < num_args; ++i) {
    TF_ASSIGN_OR_RETURN(
        ShardArgResult sharded_arg,
        ShardArg(arguments.flat_dynamic_args[i], input_devices, input_specs[i],
                 cache_entry.py_devices, python_shard_arg_fallback_));

    std::vector<xla::PjRtBuffer*>& per_device_buffers =
        sharded_arg.per_device_buffers;
    for (int computation = 0; computation < num_computations; ++computation) {
      num_computation_num_args_buffers[computation][i] =
          per_device_buffers[computation];
    }
    for (auto& owned_buffer : sharded_arg.owned_buffers) {
      arguments.keep_alive.push_back(std::move(owned_buffer));
    }
    if (sharded_arg.owning_sda) {
      arguments.keep_alive_objects.push_back(std::move(sharded_arg.owning_sda));
    }
  }
#endif

#ifdef JAX_ENABLE_IFRT
  // A vector of [num_outputs].
  std::vector<tsl::RCReference<xla::ifrt::Array>> output_arrays;
  {
    py::gil_scoped_release gil_release;
    auto ifrt_executable = cache_entry.executable->ifrt_executable();
    TF_ASSIGN_OR_RETURN(
        auto result, ifrt_executable->Execute(absl::MakeSpan(num_args_arrays),
                                              cache_entry.executable->options(),
                                              /*devices=*/std::nullopt));
    output_arrays = std::move(result.outputs);
  }
#else
  // A vector of [num_devices, num_outputs].
  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> output_buffers;
  {
    py::gil_scoped_release gil_release;
    auto pjrt_executable = cache_entry.executable->pjrt_executable();
    TF_ASSIGN_OR_RETURN(output_buffers, pjrt_executable->Execute(
                                            num_computation_num_args_buffers,
                                            cache_entry.executable->options()));
  }
#endif

  // TODO(jblespiau): We don't need to create the PyBuffer objects.
  // Having a C++ `ShardedDeviceArray`, keeping internally the PjRtBuffer
  // objects is sufficient, and we can lazily create the `PyBuffer` only if
  // we access them from Python.
  auto traceback = xla::Traceback::Get();
  // TODO(jblespiau): Change the `client` function to return a reference.
  std::shared_ptr<xla::PyClient> client = cache_entry.executable->client();

  // Convert the PjRtBuffer objects to PyBuffer, and invert the order from
  // [num_devices, num_args] to [num_args, num_devices].
#ifdef JAX_ENABLE_IFRT
  const int num_outputs = output_arrays.size();
#else
  const int num_outputs = output_buffers[0].size();
#endif
  std::vector<py::object> flat_sharded_device_arrays;
  flat_sharded_device_arrays.reserve(num_outputs);

  const auto& output_specs = cache_entry.out_result_specs;

  if (!cache_entry.out_array_shardings.empty()) {
#ifdef JAX_ENABLE_IFRT
    for (int i = 0; i < num_outputs; ++i) {
      const ResultSpec& result_spec = output_specs[i];
      xla::PyArray py_array(
          result_spec.out_aval, result_spec.weak_type,
          cache_entry.out_dtypes[i], cache_entry.out_shapes[i],
          cache_entry.out_array_shardings[i], client, traceback,
          std::move(output_arrays[i]), cache_entry.out_committed[i]);

      flat_sharded_device_arrays.push_back(std::move(py_array));
    }
#else
    for (int i = 0; i < num_outputs; ++i) {
      std::vector<std::shared_ptr<xla::PjRtBuffer>> outputs;
      outputs.reserve(num_computations);
      for (int j = 0; j < num_computations; ++j) {
        outputs.push_back(std::move(output_buffers[j][i]));
      }

      const ResultSpec& result_spec = output_specs[i];

      xla::PyArray py_array(
          result_spec.out_aval, result_spec.weak_type,
          cache_entry.out_dtypes[i], cache_entry.out_shapes[i],
          cache_entry.out_array_shardings[i], client, traceback,
          std::move(outputs), cache_entry.out_committed[i]);

      flat_sharded_device_arrays.push_back(std::move(py_array));
    }
#endif
  } else {
#ifdef JAX_ENABLE_IFRT
    std::vector<std::vector<xla::PyBuffer::object>> outputs;
    outputs.resize(num_outputs);
    for (int output_id = 0; output_id < num_outputs; ++output_id) {
      outputs[output_id].reserve(num_computations);
      TF_ASSIGN_OR_RETURN(
          auto single_device_arrays,
          output_arrays[output_id]->DisassembleIntoSingleDeviceArrays(
              xla::ifrt::ArrayCopySemantics::kReuseInput));
      for (auto& single_device_array : single_device_arrays) {
        outputs[output_id].push_back(xla::PyBuffer::Make(
            client, std::move(single_device_array), traceback));
      }
    }
#else
    std::vector<std::vector<xla::PyBuffer::object>> outputs;
    outputs.resize(num_outputs);
    for (int output_id = 0; output_id < num_outputs; ++output_id) {
      outputs[output_id].reserve(num_computations);
      for (int computation = 0; computation < num_computations; ++computation) {
        outputs[output_id].push_back(xla::PyBuffer::Make(
            client, std::move(output_buffers[computation][output_id]),
            traceback));
      }
    }
#endif

    for (int i = 0; i < num_outputs; ++i) {
      const ResultSpec& result_spec = output_specs[i];
      flat_sharded_device_arrays.push_back(ShardedDeviceArray::Make(
          /*aval=*/result_spec.out_aval,
          /*sharding_spec=*/result_spec.out_spec,
          /*device_buffers=*/py::cast(std::move(outputs[i])),
          /*indices=*/result_spec.out_indices,
          /*weak_type=*/result_spec.weak_type));
    }
  }

  py::object out =
      cache_entry.out_pytree_def.Unflatten(flat_sharded_device_arrays);

  // If there is a post-hook function, call it with the inputs and the outputs.
  std::optional<py::object> post_hook = GetPostHook();
  if (post_hook) {
    py::tuple args_tuple(num_positional_args);
    for (size_t i = 0; i < num_positional_args; ++i) {
      args_tuple[i] = args[i];
    }
