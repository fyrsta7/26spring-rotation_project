  LOG(INFO) << "Restore Op = " << item.restore_op;
  LOG(INFO) << "save_restore_loc_tensor = " << item.save_restore_loc_tensor;
  if (!item.keep_ops.empty()) {
    LOG(INFO) << offset << "keep ops  :";
    for (const auto& f : item.keep_ops) {
      LOG(INFO) << offset2 << f;
    }
  } else {
    LOG(INFO) << offset << "No keep ops";
  }
}

Status TRTOptimizationPass::Optimize(grappler::Cluster* cluster,
                                     const grappler::GrapplerItem& item,
                                     GraphDef* optimized_graph) {
  VLOG(1) << "Called TRTOptimization Pass " << name_
          << " on a grappler item with id=" << item.id;
  TF_ASSIGN_OR_RETURN(bool do_function_conversion, ShouldConvertFunction(item));
  // Optimizing the main graph(identified with `item.id == "tf_graph"`) with
  // `minimim_segment_size == -1` indicates skipping main graph conversion.
  if ((minimum_segment_size_ == -1 && item.id == "tf_graph") ||
      (item.id != "tf_graph" && !do_function_conversion)) {
    VLOG(1) << "Not optimizing this grappler item: " << item.id;
    *optimized_graph = item.graph;
    return Status::OK();
  }
  if (VLOG_IS_ON(3)) {
    LOG(INFO) << CurrentStackTrace();
    PrintDebugInfo(cluster, item);
  }

  if (use_calibration_ && precision_mode_ != TrtPrecisionMode::INT8) {
    VLOG(1) << "Calibration with FP32 or FP16 is not implemented. "
            << "Falling back to use_calibration = False."
            << "Note that the default value of use_calibration is True.";
    use_calibration_ = false;
  }

  std::vector<string> nodes_to_preserve;
  auto _nodes_to_preserve = item.NodesToPreserve();
  nodes_to_preserve.reserve(_nodes_to_preserve.size());
  for (const auto& n : _nodes_to_preserve) {
    auto tokens = str_util::Split(n, ":");
    string s = tokens.at(0);
    for (int i = 1; i < tokens.size() - 1; ++i) {
      StrAppend(&s, ":", tokens.at(i));
    }
    int dumm_port = -1;
    // If the last token is not an integer, it must be part of the name.
    // Otherwise it is port number.
    if (tokens.size() > 1 &&
        !strings::safe_strto32(tokens.back(), &dumm_port)) {  // non-absl ok
      StrAppend(&s, ":", tokens.back());
    }
    nodes_to_preserve.push_back(s);
  }

  ConversionParams cp;
  cp.grappler_item = &item;
  cp.output_names = &nodes_to_preserve;
  cp.trt_logger_name = trt_logger_name_;
  cp.max_batch_size = maximum_batch_size_;
  cp.max_workspace_size_bytes = max_workspace_size_bytes_;
  cp.output_graph_def = optimized_graph;
  cp.precision_mode = precision_mode_;
  cp.minimum_segment_size = minimum_segment_size_;
  cp.cluster = cluster;
  cp.is_dyn_op = is_dynamic_op_;
  cp.max_cached_engines = max_cached_batches_;
  cp.use_calibration = use_calibration_;
  cp.use_implicit_batch = use_implicit_batch_;
  cp.profile_strategy = profile_strategy_;
  cp.allow_build_at_runtime = allow_build_at_runtime_;

