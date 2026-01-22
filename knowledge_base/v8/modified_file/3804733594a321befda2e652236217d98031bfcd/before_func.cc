BUILTIN(HandleApiCall) {
  HandleScope scope(isolate);
  Handle<Object> receiver = args.receiver();
  Handle<HeapObject> new_target = args.new_target();
  Handle<FunctionTemplateInfo> fun_data(
      args.target()->shared().get_api_func_data(), isolate);
  int argc = args.length() - 1;
  Address* argv = args.address_of_first_argument();
  if (new_target->IsJSReceiver()) {
    RETURN_RESULT_OR_FAILURE(
        isolate, HandleApiCallHelper<true>(isolate, new_target, fun_data,
                                           receiver, argv, argc));
  } else {
    RETURN_RESULT_OR_FAILURE(
        isolate, HandleApiCallHelper<false>(isolate, new_target, fun_data,
                                            receiver, argv, argc));
  }
}
