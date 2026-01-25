// Allows the host to be a raw IP (either v4 or v6).
Status ValidateHostPortPair(const string& host_port) {
  string bns_prefix = "/bns/";
  if (host_port.substr(0, bns_prefix.length()) == bns_prefix) {
    return Status::OK();
  }
  uint32 port;
  auto colon_index = host_port.find_last_of(':');
  if (!strings::safe_strtou32(host_port.substr(colon_index + 1), &port) ||
      host_port.substr(0, colon_index).find("/") != string::npos) {
    return errors::InvalidArgument("Could not interpret \"", host_port,
                                   "\" as a host-port pair.");
  }
  return Status::OK();
}