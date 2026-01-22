void Debug::DeoptimizeFunction(Handle<SharedFunctionInfo> shared) {
  // Deoptimize all code compiled from this shared function info including
  // inlining.
  isolate_->AbortConcurrentOptimization(BlockingBehavior::kBlock);

  // TODO(mlippautz): Try to remove this call.
  isolate_->heap()->PreciseCollectAllGarbage(
      Heap::kNoGCFlags, GarbageCollectionReason::kDebugger);

  bool found_something = false;
  Code::OptimizedCodeIterator iterator(isolate_);
  do {
    Code code = iterator.Next();
    if (code.is_null()) break;
    if (code.Inlines(*shared)) {
      code.set_marked_for_deoptimization(true);
      found_something = true;
    }
  } while (true);

  if (found_something) {
    // Only go through with the deoptimization if something was found.
    Deoptimizer::DeoptimizeMarkedCode(isolate_);
  }
}
