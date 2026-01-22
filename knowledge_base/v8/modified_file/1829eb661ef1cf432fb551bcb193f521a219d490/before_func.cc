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
    if (!context_.IsEmpty()) context_->Enter();
    if (do_callback_) isolate_->FireBeforeCallEnteredCallback();
  }
