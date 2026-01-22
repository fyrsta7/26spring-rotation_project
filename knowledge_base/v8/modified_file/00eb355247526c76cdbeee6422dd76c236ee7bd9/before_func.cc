void Context::AddOptimizedFunction(JSFunction* function) {
  ASSERT(IsGlobalContext());
#ifdef DEBUG
  Object* element = get(OPTIMIZED_FUNCTIONS_LIST);
  while (!element->IsUndefined()) {
    CHECK(element != function);
    element = JSFunction::cast(element)->next_function_link();
  }

  CHECK(function->next_function_link()->IsUndefined());

  // Check that the context belongs to the weak global contexts list.
  bool found = false;
  Object* context = GetHeap()->global_contexts_list();
  while (!context->IsUndefined()) {
    if (context == this) {
      found = true;
      break;
    }
    context = Context::cast(context)->get(Context::NEXT_CONTEXT_LINK);
  }
  CHECK(found);
#endif
  function->set_next_function_link(get(OPTIMIZED_FUNCTIONS_LIST));
  set(OPTIMIZED_FUNCTIONS_LIST, function);
}
