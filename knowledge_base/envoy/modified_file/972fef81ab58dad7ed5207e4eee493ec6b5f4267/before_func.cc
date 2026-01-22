bool Utility::isUpgrade(const RequestOrResponseHeaderMap& headers) {
  // In firefox the "Connection" request header value is "keep-alive, Upgrade",
  // we should check if it contains the "Upgrade" token.
  return (headers.Upgrade() &&
          Envoy::StringUtil::caseFindToken(headers.getConnectionValue(), ",",
                                           Http::Headers::get().ConnectionValues.Upgrade.c_str()));
}
