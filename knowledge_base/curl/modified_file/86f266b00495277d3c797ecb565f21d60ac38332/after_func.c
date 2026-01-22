
  /* RFC2818 checks */
  if(found_subject_alt_names && !found_subject_alt_name_matching_conn) {
    if(data->set.ssl.verifyhost) {
      /* Break connection ! */
      Curl_axtls_close(conn, sockindex);
      failf(data, "\tsubjectAltName(s) do not match %s\n",
            conn->host.dispname);
      return CURLE_PEER_FAILED_VERIFICATION;
    }
    else
      infof(data, "\tsubjectAltName(s) do not match %s\n",
            conn->host.dispname);
  }
  else if(found_subject_alt_names == 0) {
    /* Per RFC2818, when no Subject Alt Names were available, examine the peer
       CN as a legacy fallback */
    peer_CN = ssl_get_cert_dn(ssl, SSL_X509_CERT_COMMON_NAME);
    if(peer_CN == NULL) {
      if(data->set.ssl.verifyhost) {
        Curl_axtls_close(conn, sockindex);
        failf(data, "unable to obtain common name from peer certificate");
        return CURLE_PEER_FAILED_VERIFICATION;
      }
      else
        infof(data, "unable to obtain common name from peer certificate");
    }
    else {
      if(!Curl_cert_hostcheck((const char *)peer_CN, conn->host.name)) {
        if(data->set.ssl.verifyhost) {
          /* Break connection ! */
          Curl_axtls_close(conn, sockindex);
          failf(data, "\tcommon name \"%s\" does not match \"%s\"\n",
                peer_CN, conn->host.dispname);
          return CURLE_PEER_FAILED_VERIFICATION;
        }
        else
          infof(data, "\tcommon name \"%s\" does not match \"%s\"\n",
                peer_CN, conn->host.dispname);
      }
    }
  }

  /* General housekeeping */
  conn->ssl[sockindex].state = ssl_connection_complete;
  conn->recv[sockindex] = axtls_recv;
  conn->send[sockindex] = axtls_send;

  /* Put our freshly minted SSL session in cache */
  ssl_idsize = ssl_get_session_id_size(ssl);
  ssl_sessionid = ssl_get_session_id(ssl);
  if(Curl_ssl_addsessionid(conn, (void *) ssl_sessionid, ssl_idsize)
     != CURLE_OK)
    infof (data, "failed to add session to cache\n");

  return CURLE_OK;
}
