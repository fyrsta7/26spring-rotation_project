void Pipeline::GenerateCodeForWasmFunction(
    OptimizedCompilationInfo* info, wasm::CompilationEnv* env,
    const wasm::WireBytesStorage* wire_bytes_storage, MachineGraph* mcgraph,
    CallDescriptor* call_descriptor, SourcePositionTable* source_positions,
    NodeOriginTable* node_origins, wasm::FunctionBody function_body,
    const wasm::WasmModule* module, int function_index,
    std::vector<compiler::WasmLoopInfo>* loop_info,
    wasm::AssemblerBufferCache* buffer_cache) {
  auto* wasm_engine = wasm::GetWasmEngine();
  base::TimeTicks start_time;
  if (V8_UNLIKELY(FLAG_trace_wasm_compilation_times)) {
    start_time = base::TimeTicks::Now();
  }
  ZoneStats zone_stats(wasm_engine->allocator());
  std::unique_ptr<PipelineStatistics> pipeline_statistics(
      CreatePipelineStatistics(function_body, module, info, &zone_stats));
  PipelineData data(&zone_stats, wasm_engine, info, mcgraph,
                    pipeline_statistics.get(), source_positions, node_origins,
                    WasmAssemblerOptions(), buffer_cache);

  PipelineImpl pipeline(&data);

  if (data.info()->trace_turbo_json() || data.info()->trace_turbo_graph()) {
    CodeTracer::StreamScope tracing_scope(data.GetCodeTracer());
    tracing_scope.stream()
        << "---------------------------------------------------\n"
        << "Begin compiling method " << data.info()->GetDebugName().get()
        << " using TurboFan" << std::endl;
  }

  pipeline.RunPrintAndVerify("V8.WasmMachineCode", true);

  data.BeginPhaseKind("V8.WasmOptimization");
  if (FLAG_wasm_inlining) {
    pipeline.Run<WasmInliningPhase>(env, function_index, wire_bytes_storage,
                                    loop_info);
    pipeline.RunPrintAndVerify(WasmInliningPhase::phase_name(), true);
  }
  if (FLAG_wasm_loop_peeling) {
    pipeline.Run<WasmLoopPeelingPhase>(loop_info);
    pipeline.RunPrintAndVerify(WasmLoopPeelingPhase::phase_name(), true);
  }
  if (FLAG_wasm_loop_unrolling) {
    pipeline.Run<WasmLoopUnrollingPhase>(loop_info);
    pipeline.RunPrintAndVerify(WasmLoopUnrollingPhase::phase_name(), true);
  }
  const bool is_asm_js = is_asmjs_module(module);

  if (FLAG_experimental_wasm_gc || FLAG_experimental_wasm_stringref) {
    pipeline.Run<WasmTypingPhase>(function_index);
    pipeline.RunPrintAndVerify(WasmTypingPhase::phase_name(), true);
    if (FLAG_wasm_opt) {
      pipeline.Run<WasmGCOptimizationPhase>(module);
      pipeline.RunPrintAndVerify(WasmGCOptimizationPhase::phase_name(), true);
    }
    pipeline.Run<WasmGCLoweringPhase>();
    pipeline.RunPrintAndVerify(WasmGCLoweringPhase::phase_name(), true);
  }

  if (FLAG_wasm_opt || is_asm_js) {
    pipeline.Run<WasmOptimizationPhase>(is_asm_js);
    pipeline.RunPrintAndVerify(WasmOptimizationPhase::phase_name(), true);
  } else {
    pipeline.Run<WasmBaseOptimizationPhase>();
    pipeline.RunPrintAndVerify(WasmBaseOptimizationPhase::phase_name(), true);
  }

  pipeline.Run<MemoryOptimizationPhase>();
  pipeline.RunPrintAndVerify(MemoryOptimizationPhase::phase_name(), true);

  if (FLAG_experimental_wasm_gc && FLAG_wasm_opt) {
    // Run value numbering and machine operator reducer to optimize load/store
    // address computation (in particular, reuse the address computation
    // whenever possible).
    pipeline.Run<MachineOperatorOptimizationPhase>();
    pipeline.RunPrintAndVerify(MachineOperatorOptimizationPhase::phase_name(),
                               true);
    pipeline.Run<DecompressionOptimizationPhase>();
    pipeline.RunPrintAndVerify(DecompressionOptimizationPhase::phase_name(),
                               true);
  }

  if (FLAG_wasm_opt) {
    pipeline.Run<BranchConditionDuplicationPhase>();
    pipeline.RunPrintAndVerify(BranchConditionDuplicationPhase::phase_name(),
                               true);
  }

  if (FLAG_turbo_splitting && !is_asm_js) {
    data.info()->set_splitting();
  }

  if (data.node_origins()) {
    data.node_origins()->RemoveDecorator();
  }

  data.BeginPhaseKind("V8.InstructionSelection");
  pipeline.ComputeScheduledGraph();

  Linkage linkage(call_descriptor);
  if (!pipeline.SelectInstructions(&linkage)) return;
  pipeline.AssembleCode(&linkage);

  auto result = std::make_unique<wasm::WasmCompilationResult>();
  CodeGenerator* code_generator = pipeline.code_generator();
  code_generator->tasm()->GetCode(
      nullptr, &result->code_desc, code_generator->safepoint_table_builder(),
      static_cast<int>(code_generator->handler_table_offset()));

  result->instr_buffer = code_generator->tasm()->ReleaseBuffer();
  result->frame_slot_count = code_generator->frame()->GetTotalFrameSlotCount();
  result->tagged_parameter_slots = call_descriptor->GetTaggedParameterSlots();
  result->source_positions = code_generator->GetSourcePositionTable();
  result->protected_instructions_data =
      code_generator->GetProtectedInstructionsData();
  result->result_tier = wasm::ExecutionTier::kTurbofan;

  if (data.info()->trace_turbo_json()) {
    TurboJsonFile json_of(data.info(), std::ios_base::app);
    json_of << "{\"name\":\"disassembly\",\"type\":\"disassembly\""
            << BlockStartsAsJSON{&code_generator->block_starts()}
            << "\"data\":\"";
#ifdef ENABLE_DISASSEMBLER
    std::stringstream disassembler_stream;
    Disassembler::Decode(
        nullptr, disassembler_stream, result->code_desc.buffer,
        result->code_desc.buffer + result->code_desc.safepoint_table_offset,
        CodeReference(&result->code_desc));
    for (auto const c : disassembler_stream.str()) {
      json_of << AsEscapedUC16ForJSON(c);
    }
#endif  // ENABLE_DISASSEMBLER
    json_of << "\"}\n]";
    json_of << "\n}";
  }

  if (data.info()->trace_turbo_json() || data.info()->trace_turbo_graph()) {
    CodeTracer::StreamScope tracing_scope(data.GetCodeTracer());
    tracing_scope.stream()
        << "---------------------------------------------------\n"
        << "Finished compiling method " << data.info()->GetDebugName().get()
        << " using TurboFan" << std::endl;
  }

  if (V8_UNLIKELY(FLAG_trace_wasm_compilation_times)) {
    base::TimeDelta time = base::TimeTicks::Now() - start_time;
    int codesize = result->code_desc.body_size();
    StdoutStream{} << "Compiled function "
                   << reinterpret_cast<const void*>(module) << "#"
                   << function_index << " using TurboFan, took "
                   << time.InMilliseconds() << " ms and "
                   << zone_stats.GetMaxAllocatedBytes() << " / "
                   << zone_stats.GetTotalAllocatedBytes()
                   << " max/total bytes; bodysize "
                   << function_body.end - function_body.start << " codesize "
                   << codesize << " name " << data.info()->GetDebugName().get()
                   << std::endl;
  }

  DCHECK(result->succeeded());
  info->SetWasmCompilationResult(std::move(result));
}
