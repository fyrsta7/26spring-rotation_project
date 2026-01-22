  return result;
}

envoy_cert_validation_result verifyX509CertChain(const std::vector<std::string>& certs,
                                                 absl::string_view hostname) {
  CertVerifyStatus result;
  bool is_issued_by_known_root;
  std::vector<std::string> verified_chain;
  std::vector<std::string> cert_chain;
  cert_chain.reserve(certs.size());
  for (absl::string_view cert : certs) {
    cert_chain.push_back(std::string(cert));
  }

  // Android ignores the authType parameter to X509TrustManager.checkServerTrusted, so pass in "RSA"
  // as dummy value. See https://crbug.com/627154.
  jvmVerifyX509CertChain(cert_chain, "RSA", hostname, &result, &is_issued_by_known_root,
                         &verified_chain);
  switch (result) {
  case CertVerifyStatus::Ok:
    return {ENVOY_SUCCESS, 0, nullptr};
  case CertVerifyStatus::Expired: {
    return {ENVOY_FAILURE, SSL_AD_CERTIFICATE_EXPIRED,
            "AndroidNetworkLibrary_verifyServerCertificates failed: expired cert."};
  }
  case CertVerifyStatus::NoTrustedRoot:
    return {ENVOY_FAILURE, SSL_AD_CERTIFICATE_UNKNOWN,
            "AndroidNetworkLibrary_verifyServerCertificates failed: no trusted root."};
  case CertVerifyStatus::UnableToParse:
    return {ENVOY_FAILURE, SSL_AD_BAD_CERTIFICATE,
            "AndroidNetworkLibrary_verifyServerCertificates failed: unable to parse cert."};
  case CertVerifyStatus::IncorrectKeyUsage:
    return {ENVOY_FAILURE, SSL_AD_CERTIFICATE_UNKNOWN,
            "AndroidNetworkLibrary_verifyServerCertificates failed: incorrect key usage."};
  case CertVerifyStatus::Failed:
    return {
        ENVOY_FAILURE, SSL_AD_CERTIFICATE_UNKNOWN,
        "AndroidNetworkLibrary_verifyServerCertificates failed: validation couldn't be conducted."};
  case CertVerifyStatus::NotYetValid:
    return {ENVOY_FAILURE, SSL_AD_CERTIFICATE_UNKNOWN,
            "AndroidNetworkLibrary_verifyServerCertificates failed: not yet valid."};
  default:
