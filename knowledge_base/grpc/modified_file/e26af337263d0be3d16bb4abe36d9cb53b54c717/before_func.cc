  if (grpc_error_get_int(error, which, &unused)) {
    return error;
  }
  if (grpc_error_is_special(error)) return nullptr;
  // Otherwise, search through its children.
  uint8_t slot = error->first_err;
  while (slot != UINT8_MAX) {
    grpc_linked_error* lerr =
        reinterpret_cast<grpc_linked_error*>(error->arena + slot);
    grpc_error* result = recursively_find_error_with_field(lerr->err, which);
    if (result) return result;
    slot = lerr->next;
  }
  return nullptr;
}

void grpc_error_get_status(grpc_error* error, grpc_millis deadline,
                           grpc_status_code* code, grpc_slice* slice,
                           grpc_http2_error_code* http_error,
                           const char** error_string) {
  // Start with the parent error and recurse through the tree of children
  // until we find the first one that has a status code.
  grpc_error* found_error =
      recursively_find_error_with_field(error, GRPC_ERROR_INT_GRPC_STATUS);
  if (found_error == nullptr) {
    /// If no grpc-status exists, retry through the tree to find a http2 error
    /// code
    found_error =
        recursively_find_error_with_field(error, GRPC_ERROR_INT_HTTP2_ERROR);
  }

  // If we found an error with a status code above, use that; otherwise,
  // fall back to using the parent error.
  if (found_error == nullptr) found_error = error;

  grpc_status_code status = GRPC_STATUS_UNKNOWN;
  intptr_t integer;
  if (grpc_error_get_int(found_error, GRPC_ERROR_INT_GRPC_STATUS, &integer)) {
    status = static_cast<grpc_status_code>(integer);
  } else if (grpc_error_get_int(found_error, GRPC_ERROR_INT_HTTP2_ERROR,
                                &integer)) {
    status = grpc_http2_error_to_grpc_status(
        static_cast<grpc_http2_error_code>(integer), deadline);
  }
  if (code != nullptr) *code = status;

  if (error_string != nullptr && status != GRPC_STATUS_OK) {
    *error_string = gpr_strdup(grpc_error_string(error));
  }

  if (http_error != nullptr) {
    if (grpc_error_get_int(found_error, GRPC_ERROR_INT_HTTP2_ERROR, &integer)) {
      *http_error = static_cast<grpc_http2_error_code>(integer);
    } else if (grpc_error_get_int(found_error, GRPC_ERROR_INT_GRPC_STATUS,
                                  &integer)) {
      *http_error =
          grpc_status_to_http2_error(static_cast<grpc_status_code>(integer));
