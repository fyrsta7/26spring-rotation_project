ZoneUnorderedSet<Node*>* LoopFinder::FindSmallInnermostLoopFromHeader(
    Node* loop_header, AllNodes& all_nodes, Zone* zone, size_t max_size,
    Purpose purpose) {
  auto* visited = zone->New<ZoneUnorderedSet<Node*>>(zone);
  std::vector<Node*> queue;

  DCHECK_EQ(loop_header->opcode(), IrOpcode::kLoop);

  queue.push_back(loop_header);

#define ENQUEUE_USES(use_name, condition)                                      \
  for (Node * use_name : node->uses()) {                                       \
    if (condition && visited->count(use_name) == 0) queue.push_back(use_name); \
  }
  bool has_instruction_worth_peeling = false;

  while (!queue.empty()) {
    Node* node = queue.back();
    queue.pop_back();
    if (node->opcode() == IrOpcode::kEnd) {
      // We reached the end of the graph. The end node is not part of the loop.
      continue;
    }
    visited->insert(node);
    if (visited->size() > max_size) return nullptr;
    switch (node->opcode()) {
      case IrOpcode::kLoop:
        // Found nested loop.
        if (node != loop_header) return nullptr;
        ENQUEUE_USES(use, true);
        break;
      case IrOpcode::kLoopExit:
        // Found nested loop.
        if (node->InputAt(1) != loop_header) return nullptr;
        // LoopExitValue/Effect uses are inside the loop. The rest are not.
        ENQUEUE_USES(use, (use->opcode() == IrOpcode::kLoopExitEffect ||
                           use->opcode() == IrOpcode::kLoopExitValue))
        break;
      case IrOpcode::kLoopExitEffect:
      case IrOpcode::kLoopExitValue:
        if (NodeProperties::GetControlInput(node)->InputAt(1) != loop_header) {
          // Found nested loop.
          return nullptr;
        }
        // All uses are outside the loop, do nothing.
        break;
      // If unrolling, call nodes are considered to have unbounded size,
      // i.e. >max_size, with the exception of certain wasm builtins.
      case IrOpcode::kTailCall:
      case IrOpcode::kJSWasmCall:
      case IrOpcode::kJSCall:
        if (purpose == Purpose::kLoopUnrolling) return nullptr;
        ENQUEUE_USES(use, true)
        break;
      case IrOpcode::kCall: {
        if (purpose == Purpose::kLoopPeeling) {
          ENQUEUE_USES(use, true);
          break;
        }
        Node* callee = node->InputAt(0);
        if (callee->opcode() != IrOpcode::kRelocatableInt32Constant &&
            callee->opcode() != IrOpcode::kRelocatableInt64Constant) {
          return nullptr;
        }
        intptr_t info =
            OpParameter<RelocatablePtrConstantInfo>(callee->op()).value();
        using WasmCode = v8::internal::wasm::WasmCode;
        constexpr intptr_t unrollable_builtins[] = {
            // Exists in every stack check.
            WasmCode::kWasmStackGuard,
            // Fast table operations.
            WasmCode::kWasmTableGet, WasmCode::kWasmTableSet,
            WasmCode::kWasmTableGetFuncRef, WasmCode::kWasmTableSetFuncRef,
            WasmCode::kWasmTableGrow,
            // Atomics.
            WasmCode::kWasmAtomicNotify, WasmCode::kWasmI32AtomicWait,
            WasmCode::kWasmI64AtomicWait,
            // Exceptions.
            WasmCode::kWasmAllocateFixedArray, WasmCode::kWasmThrow,
            WasmCode::kWasmRethrow, WasmCode::kWasmRethrowExplicitContext,
            // Fast wasm-gc operations.
            WasmCode::kWasmRefFunc};
        if (std::count(std::begin(unrollable_builtins),
                       std::end(unrollable_builtins), info) == 0) {
          return nullptr;
        }
        ENQUEUE_USES(use, true)
        break;
      }
      case IrOpcode::kWasmStructGet: {
        // When a chained load occurs in the loop, assume that peeling might
        // help.
        // Extending this idea to array.get/array.len has been found to hurt
        // more than it helps (tested on Sheets, Feb 2023).
        Node* object = node->InputAt(0);
        if (object->opcode() == IrOpcode::kWasmStructGet &&
            visited->find(object) != visited->end()) {
          has_instruction_worth_peeling = true;
        }
        ENQUEUE_USES(use, true);
        break;
      }
      case IrOpcode::kStringPrepareForGetCodeunit:
        has_instruction_worth_peeling = true;
        V8_FALLTHROUGH;
      default:
        ENQUEUE_USES(use, true)
        break;
    }
  }

  // Check that there is no floating control other than direct nodes to start().
  // We do this by checking that all non-start control inputs of loop nodes are
  // also in the loop.
  // TODO(manoskouk): This is a safety check. Consider making it DEBUG-only when
  // we are confident there is no incompatible floating control generated in
  // wasm.
  for (Node* node : *visited) {
    // The loop header is allowed to point outside the loop.
    if (node == loop_header) continue;

    if (!all_nodes.IsLive(node)) continue;

    for (Edge edge : node->input_edges()) {
      Node* input = edge.to();
      if (NodeProperties::IsControlEdge(edge) && visited->count(input) == 0 &&
          input->opcode() != IrOpcode::kStart) {
        FATAL(
            "Floating control detected in wasm turbofan graph: Node #%d:%s is "
            "inside loop headed by #%d, but its control dependency #%d:%s is "
            "outside",
            node->id(), node->op()->mnemonic(), loop_header->id(), input->id(),
            input->op()->mnemonic());
      }
    }
  }

  // Only peel functions containing instructions for which loop peeling is known
  // to be useful. TODO(7748): Add more instructions to get more benefits out of
  // loop peeling.
  if (purpose == Purpose::kLoopPeeling && !has_instruction_worth_peeling) {
    return nullptr;
  }
  return visited;
}
