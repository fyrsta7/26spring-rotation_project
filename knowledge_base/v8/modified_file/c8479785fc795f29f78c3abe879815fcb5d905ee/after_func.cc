static Object* ComputeReceiverForNonGlobal(Isolate* isolate, JSObject* holder) {
  DCHECK(!holder->IsGlobalObject());

  // If the holder isn't a context extension object, we just return it
  // as the receiver. This allows arguments objects to be used as
  // receivers, but only if they are put in the context scope chain
  // explicitly via a with-statement.
  if (holder->map()->instance_type() != JS_CONTEXT_EXTENSION_OBJECT_TYPE) {
    return holder;
  }
  // Fall back to using the global object as the implicit receiver if
  // the property turns out to be a local variable allocated in a
  // context extension object - introduced via eval.
  return isolate->heap()->undefined_value();
}
