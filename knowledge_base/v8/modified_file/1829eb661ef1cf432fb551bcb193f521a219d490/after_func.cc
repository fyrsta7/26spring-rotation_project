  explicit CallDepthScope(i::Isolate* isolate, Local<Context> context,
                          bool do_callback)
      : isolate_(isolate),
        context_(context),
        escaped_(false),
        do_callback_(do_callback) {
    // TODO(dcarney): remove this when blink stops crashing.
    DCHECK(!isolate_->external_caught_exception());
    isolate_->IncrementJsCallsFromApiCounter();
    isolate_->handle_scope_implementer()->IncrementCallDepth();
    if (!context.IsEmpty()) {
      i::Handle<i::Context> env = Utils::OpenHandle(*context);
      i::HandleScopeImplementer* impl = isolate->handle_scope_implementer();
      if (isolate->context() != nullptr &&
          isolate->context()->native_context() == env->native_context() &&
          impl->LastEnteredContextWas(env)) {
        context_ = Local<Context>();
      } else {
        context_->Enter();
      }
    }
    if (do_callback_) isolate_->FireBeforeCallEnteredCallback();
  }
