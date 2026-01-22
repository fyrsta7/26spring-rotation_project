  if (retries == kRetriesBound) {
    throw std::runtime_error("Hits microtasks retries bound.");
  }
}

void JSIExecutor::callFunction(
    const std::string &moduleId,
    const std::string &methodId,
    const folly::dynamic &arguments) {
  SystraceSection s(
      "JSIExecutor::callFunction", "moduleId", moduleId, "methodId", methodId);
  if (!callFunctionReturnFlushedQueue_) {
    bindBridge();
  }

  // Construct the error message producer in case this times out.
  // This is executed on a background thread, so it must capture its parameters
  // by value.
  auto errorProducer = [=] {
    std::stringstream ss;
    ss << "moduleID: " << moduleId << " methodID: " << methodId
       << " arguments: " << folly::toJson(arguments);
    return ss.str();
  };

  Value ret = Value::undefined();
  try {
    scopedTimeoutInvoker_(
        [&] {
          ret = callFunctionReturnFlushedQueue_->call(
              *runtime_,
              moduleId,
              methodId,
              valueFromDynamic(*runtime_, arguments));
        },
        std::move(errorProducer));
  } catch (...) {
    std::throw_with_nested(
        std::runtime_error("Error calling " + moduleId + "." + methodId));
  }
