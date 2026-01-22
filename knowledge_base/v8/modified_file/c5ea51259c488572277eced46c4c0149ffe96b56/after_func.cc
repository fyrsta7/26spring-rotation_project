  void Run(PipelineData* data, Zone* temp_zone) {
    if (!data->info()->is_optimizing_from_bytecode()) {
      AstLoopAssignmentAnalyzer analyzer(data->graph_zone(), data->info());
      LoopAssignmentAnalysis* loop_assignment = analyzer.Analyze();
      data->set_loop_assignment(loop_assignment);
    }
  }
