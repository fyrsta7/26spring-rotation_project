ResponsePtr RawHttpClientImpl::toResponse(Http::ResponseMessagePtr message) {
  const uint64_t status_code = Http::Utility::getResponseStatus(message->headers());

  // Set an error status if the call to the authorization server returns any of the 5xx HTTP error
  // codes. A Forbidden response is sent to the client if the filter has not been configured with
  // failure_mode_allow.
  if (Http::CodeUtility::is5xx(status_code)) {
    return std::make_unique<Response>(errorResponse());
  }

  // Extract headers-to-remove from the storage header coming from the
  // authorization server.
  const auto& storage_header_name = Headers::get().EnvoyAuthHeadersToRemove;
  // If we are going to construct an Ok response we need to save the
  // headers_to_remove in a variable first.
  std::vector<Http::LowerCaseString> headers_to_remove;
  if (status_code == enumToInt(Http::Code::OK)) {
    const auto& get_result = message->headers().getAll(storage_header_name);
    for (size_t i = 0; i < get_result.size(); ++i) {
      const Http::HeaderEntry* entry = get_result[i];
      if (entry != nullptr) {
        absl::string_view storage_header_value = entry->value().getStringView();
        std::vector<absl::string_view> header_names = StringUtil::splitToken(
            storage_header_value, ",", /*keep_empty_string=*/false, /*trim_whitespace=*/true);
        headers_to_remove.reserve(headers_to_remove.size() + header_names.size());
        for (const auto header_name : header_names) {
          headers_to_remove.push_back(Http::LowerCaseString(std::string(header_name)));
        }
      }
    }
  }
  // Now remove the storage header from the authz server response headers before
  // we reuse them to construct an Ok/Denied authorization response below.
  message->headers().remove(storage_header_name);

  // Create an Ok authorization response.
  if (status_code == enumToInt(Http::Code::OK)) {
    SuccessResponse ok{message->headers(), config_->upstreamHeaderMatchers(),
                       config_->upstreamHeaderToAppendMatchers(),
                       Response{CheckStatus::OK, ErrorKind::Other, Http::HeaderVector{},
                                Http::HeaderVector{}, Http::HeaderVector{},
                                std::move(headers_to_remove), EMPTY_STRING, Http::Code::OK,
                                ProtobufWkt::Struct{}}};
    return std::move(ok.response_);
  }

  // Create a Denied authorization response.
  SuccessResponse denied{message->headers(), config_->clientHeaderMatchers(),
                         config_->upstreamHeaderToAppendMatchers(),
                         Response{CheckStatus::Denied,
                                  ErrorKind::Other,
                                  Http::HeaderVector{},
                                  Http::HeaderVector{},
                                  Http::HeaderVector{},
                                  {{}},
                                  message->bodyAsString(),
                                  static_cast<Http::Code>(status_code),
                                  ProtobufWkt::Struct{}}};
  return std::move(denied.response_);
}
