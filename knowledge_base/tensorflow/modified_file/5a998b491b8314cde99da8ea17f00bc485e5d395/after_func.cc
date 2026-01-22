  info.function_graph->ToGraphDef(function_graph_def);
  // Set lib_def into the function_graph.
  *function_graph_def->mutable_library() = info.lib_def.ToProto();
  *proto.mutable_ret_types() = {info.ret_types.begin(), info.ret_types.end()};
  proto.set_num_return_nodes(info.num_return_nodes);
  *proto.mutable_node_name_to_control_ret() = {
      info.node_name_to_control_ret.begin(),
      info.node_name_to_control_ret.end()};
  proto.set_optimization_time_usecs(info.optimization_duration_usecs);
  proto.set_source(info.optimization_source);
  return proto;
}

StatusOr<OptimizedFunctionGraphInfo> OptimizedFunctionGraphInfo::FromProto(
    OptimizedFunctionGraph&& proto) {
  // Reconstruct the lib_def.
  FunctionLibraryDefinition lib_def(OpRegistry::Global(),
                                    proto.function_graph().library());

  // Reconstruct the graph.
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  GraphConstructorOptions options;
  options.allow_internal_ops = true;
  options.expect_device_spec = true;
  TF_RETURN_IF_ERROR(ConvertGraphDefToGraph(
      options, std::move(*proto.mutable_function_graph()), graph.get()));

  // Clear both library and registry as the op lookup should be from lib_def.
  graph->mutable_flib_def()->set_default_registry(nullptr);
  graph->mutable_flib_def()->Clear();

