  } else {
    printer->Print(vars, "Request:($request_class$ *)request");
  }

  // TODO(jcanizales): Put this on a new line and align colons.
  if (method->server_streaming()) {
    printer->Print(vars, " eventHandler:(void(^)(BOOL done, "
      "$response_class$ *response, NSError *error))eventHandler");
  } else {
