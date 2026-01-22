  // Set attribute for tf.data functions. We cannot do this in the DFS directly
  // because `FunctionLibraryDefinition` does not seem to provide mutable access
  // to a `FunctionDef`.
  for (FunctionDef& func_def : *fdef_lib.mutable_function()) {
    const std::string& func_name = func_def.signature().name();
    if (tf_data_functions.contains(func_name) &&
        !data::IsTFDataFunction(func_def)) {
      VLOG(2) << "Marking " << func_name << " as tf.data function";
      (*func_def.mutable_attr())[data::kTFDataFunction].set_b(true);
    }
  }
}

Status MetaOptimizer::OptimizeConsumeItem(Cluster* cluster, GrapplerItem&& item,
                                          GraphDef* optimized_graph) {
  tensorflow::metrics::ScopedCounter<2> timings(
      tensorflow::metrics::GetGraphOptimizationCounter(),
      {kGrapplerCategory, "*"});

  VLOG(1) << "Starting optimization for grappler item: " << item.id;
  optimization_results_.clear();

  // Constructs a FunctionLibraryDefinition with functions that are reachable
  // from the nodes of the graph.
  const auto minimized_flib =
      [](const GraphDef& graph) -> FunctionLibraryDefinition {
    return FunctionLibraryDefinition(OpRegistry::Global(), graph.library())
        .ReachableDefinitions(graph);
  };

  // 0. Original graph might contain a huge function library, that is mostly
  // unused. This library copied over by each individual Grappler optimizer,
  // which adds a huge overhead. Before starting optimization passes we just
  // remove all the unreachable functions.
  // TODO(ezhulenev): Construct reachable function library definition directly
  // from the proto without constructing temporary FunctionLibraryDefinition.
  int old_library_size = item.graph.library().function_size();
  *item.graph.mutable_library() = minimized_flib(item.graph).ToProto();
  int new_library_size = item.graph.library().function_size();

  VLOG(1) << absl::Substitute(
      "Deleted $0 unreachable functions from the graph (library size = $1)",
      old_library_size - new_library_size, new_library_size);

  // Save a few small fields from item before we move it.
  bool optimize_function_library =
      item.optimization_options().optimize_function_library;
  const auto producer = item.graph.versions().producer();

  // 1. Optimize main graph
  TF_RETURN_IF_ERROR(OptimizeGraph(cluster, std::move(item), optimized_graph));
  VLOG(1) << "Optimized main graph.";
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

  // 2. Optimize functions reachable from the optimized graph.
  FunctionLibraryDefinition flib = minimized_flib(*optimized_graph);
  using NodeDefs = protobuf::RepeatedPtrField<NodeDef>;

  // Find functions for which we might need to compute a gradient at runtime.
  absl::flat_hash_set<string> differentiable_functions;

  const auto find_differentiable_functions =
      [&](const NodeDefs& nodes) -> void {
    for (const NodeDef& node : nodes) {
      if (IsSymbolicGradient(node)) {
        const auto* f_attr = gtl::FindOrNull(node.attr(), "f");
        if (f_attr) differentiable_functions.insert(f_attr->func().name());
      }
    }
  };

  // SymbolicGradient nodes inside the main graph.
  find_differentiable_functions(optimized_graph->node());
  // SymbolicGradient nodes inside the function library.
  for (const FunctionDef& function : optimized_graph->library().function()) {
    find_differentiable_functions(function.node_def());
  }

  // Find functions that will be compiled by XLA later
  // We do it by looking for XlaLaunch ops that call functions,
  // then depth first search down those functions to find transitive functions.
  // Grappler rewrites can potentially add nodes that are
  // not supported by XLA, so we choose to skip such functions when we optimize
  // the function library.
  absl::flat_hash_set<string> xla_compiled_functions;
  std::function<void(const string&)> find_all_functions;
  find_all_functions = [&](const string& func) -> void {
    // Ignore call cycles in the graph
    if (xla_compiled_functions.contains(func)) return;
    // Find func in the flib
    const FunctionDef* func_def = flib.Find(func);
    CHECK(func_def) << "not found: " << func;
    // Mark function to be ignored by grappler
    xla_compiled_functions.insert(func);
    // Depth first search through the func for transitively called funcs
    for (const NodeDef& node : func_def->node_def()) {
      for (const auto& attr : node.attr()) {
        const AttrValue& attr_value = attr.second;
        if (attr_value.has_func()) {
          find_all_functions(attr_value.func().name());
        }
      }
    }
  };

  auto find_xla_compiled_functions = [&](const NodeDefs& nodes) -> void {
    NameAttrList function;
    for (const NodeDef& node : nodes) {
      // Look only for XlaLaunch nodes that call a function
      if (!IsXlaLaunch(node)) continue;
      if (!GetNodeAttr(node, "function", &function).ok()) continue;
      // Find all transitively called functions
      find_all_functions(function.name());
    }
  };

  // XlaLaunch ops inside the main graph ...
  find_xla_compiled_functions(optimized_graph->node());
  // ... and inside the function library.
  for (const FunctionDef& function : optimized_graph->library().function()) {
    find_xla_compiled_functions(function.node_def());
  }
  // Propagate `_tf_data_function` attributes from functions to their callees.
  PropagateTFDataAttrs(flib, *optimized_graph->mutable_library());

  // Optimize each function only once.
  absl::flat_hash_set<string> optimized_funcs;
  while (optimize_function_library) {
    optimize_function_library = false;

    int function_idx = 0;
    for (const FunctionDef& func : optimized_graph->library().function()) {
      GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();

      const string& func_name = func.signature().name();

      // Skip functions that are not reachable from the optimized graph.
      if (!flib.Contains(func_name)) continue;
      // Skip already optimized functions.
      if (optimized_funcs.contains(func_name)) continue;
      // Skip functions that will be compiled by XLA.
      if (xla_compiled_functions.contains(func_name)) continue;

      // Skip parametrized functions (function type or body is defined only at
      // function call time by caller node attributes).
      // They should be specialized to their instantiation type parameters by
      // the function optimizer, before we can optimize function body.
      if (IsParametrized(func)) continue;

      // Skip tf.data functions as they are optimized by tf.data meta optimizer
      // and in function instantiation.
      if (data::IsTFDataFunction(func)) continue;

      VLOG(3) << "Optimize function: function=" << func_name << " ["
              << function_idx++ << " of "
              << optimized_graph->library().function_size() << "]";

      // Function optimization might specialize nested function calls, so we
      // have to reset the flag and do at least one more pass over the library.
      optimize_function_library = true;
      optimized_funcs.insert(func_name);

      // Make a GrapplerItem from a FunctionDef.
      GrapplerFunctionItem func_item;
      TF_RETURN_IF_ERROR(
          MakeGrapplerFunctionItem(func, flib, producer, &func_item));

      // If we need to compute the gradient of optimized function at runtime, we
      // can't perform non-differentiable rewrites.
      func_item.optimization_options().allow_non_differentiable_rewrites =
          !differentiable_functions.contains(func_name);

      // Device set available to the function is defined only by the runtime,
      // when we instantiate and execute the function. We can't use all devices
      // available to the main graph, because after partitioning the function
      // call node might execute on a remote worker.
      if (!func_item.devices().empty()) {
        return errors::Internal("GrapplerFunctionItem devices must be empty.");
      }

      // We are not allowed to prune certain types of ops from the graph
      // instantiated by the function definition, because we must guarantee
      // function execution semantics wrt side effects (see
      // function_optimizer.cc).
      func_item.optimization_options().allow_pruning_stateful_and_dataset_ops =
          false;

      // Optimize function body graph.
      GraphDef optimized_func_graph;
      if (IsTPUGraphDef(*optimized_graph)) {
        // Skip optimizing functions if this is a TPU graph. Currently, Grappler
        // passes do not handle TPU functions correctly in a variety of ways
        // (Note that due to the pre-placement TPU graph rewriting passes, the
        // TPU-related ops are encapsulated away into functions). For example,
        // TPU graphs contain TPUReplicateMetadata node that carries relevant
        // TPU metadata and Grappler passes could prune that away. Grappler
        // passes could also cause issues around shape inference. Since the
        // desired and existing behavior is to not optimize TPU functions with
        // Grappler, this check preserves that. The only exception is
        // implementation selector what is required to swap in some TPU specific
        // lowering code and is verified the work correctly on TPUs.
        ImplementationSelector implementation_selector;

        // Implementation selector needs to have access to valid function
        // signature and attributes, and it doesn't need actual function body.
        std::unique_ptr<FunctionDefLibrary> func_item_function_library(
            func_item.graph.release_library());
        *func_item.graph.mutable_library() =
            GetFunctionDefLibraryStub(*func_item_function_library);

        TF_RETURN_IF_ERROR(implementation_selector.Optimize(
            cluster, func_item, &optimized_func_graph));
      } else {
        GrapplerFunctionItem func_item_copy = func_item;
        TF_RETURN_IF_ERROR(OptimizeGraph(cluster, std::move(func_item_copy),
                                         &optimized_func_graph));
      }

      // Function body optimization might have created new specialized
      // functions for each instantiation context. Add them to the library.
      for (const FunctionDef& func_def :
           optimized_func_graph.library().function()) {
        if (flib.Find(func_def.signature().name()) == nullptr) {
          TF_RETURN_IF_ERROR(flib.AddFunctionDef(func_def));
        }
      }

      // Convert optimized graph back to FunctionDef.
      FunctionDef optimized_func;
      func_item.SwapFunctionBody(std::move(optimized_func_graph));
      TF_RETURN_IF_ERROR(MakeFunctionDef(func_item, flib, &optimized_func));

      // Replace optimized function with a new FunctionDef.
      TF_RETURN_IF_ERROR(flib.ReplaceFunction(func_name, optimized_func));
    }

    // If optimized at least one function, update the graph library.
    if (optimize_function_library) {
      *optimized_graph->mutable_library() = flib.ToProto();
    }
  }
