constexpr char kCeilOp[] = "Ceil";
constexpr char kBatchOp[] = "BatchDataset";
constexpr char kBatchV2Op[] = "BatchDatasetV2";
constexpr char kExperimentalMapAndBatchOp[] = "ExperimentalMapAndBatchDataset";
constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";
constexpr char kMapOp[] = "MapDataset";
constexpr char kParallelMapOp[] = "ParallelMapDataset";
constexpr char kParallelMapV2Op[] = "ParallelMapDatasetV2";
constexpr char kChooseFastestOp[] = "ChooseFastestBranchDataset";
constexpr char kPrefetchOp[] = "PrefetchDataset";

// Returns a FunctionDef containing a MapDefun op that wraps the original
// function.
FunctionDef* CreateMapDefunWrapper(const NodeDef& map_node,
                                   const FunctionDef& orig_func,
                                   FunctionDefLibrary* library) {
  FunctionDef* vectorized_func = library->add_function();
  // Function inputs and outputs are the same as original, just
  // with different shapes.
  *vectorized_func->mutable_signature() = orig_func.signature();
  graph_utils::SetUniqueGraphFunctionName("naively_vectorized_fn", library,
                                          vectorized_func);

  // Add MapDefun node
  NodeDef* map_defun_node = vectorized_func->mutable_node_def()->Add();
  map_defun_node->set_op("MapDefun");
  function_utils::SetUniqueFunctionNodeName(map_defun_node->op(),
                                            vectorized_func, map_defun_node);

  // Set attrs and inputs
  for (const string& k : {"f", "output_types", "output_shapes"}) {
    // Function, output types and (unbatched) shapes are the same as the
    // original map node.
    graph_utils::CopyAttribute(k, map_node, map_defun_node);
  }

  // Note that the inputs to the function are either regular arguments (for
  // which the function is mapped across their 0th dimension) or captured inputs
  // (for which the function takes the argument wholesale). We can infer
  // the split between these arguments from the `map_node`'s attrs.
  // The Targuments attr on `map_node` corresponds to a list of types of
  // MapDataset's captured inputs.
  auto t_captured = map_node.attr().at("Targuments");

  // Get types of input arguments from original map function
  DataTypeVector t_args;  // Regular arguments
  for (const auto& input : vectorized_func->signature().input_arg()) {
    t_args.push_back(input.type());
    map_defun_node->add_input(input.name());
  }
  // Erase the captured arguments from Targuments
  t_args.erase(t_args.end() - t_captured.list().type_size(), t_args.end());
