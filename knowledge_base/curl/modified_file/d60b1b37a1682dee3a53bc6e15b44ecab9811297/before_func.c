 * => CONNECT action.
 *
 * Set 'clear' to TRUE to have it also clear the state variable.
 */
static bool multi_ischanged(struct Curl_multi *multi, bool clear)
{
  bool retval = multi->recheckstate;
  if(clear)
    multi->recheckstate = FALSE;
  return retval;
}

CURLMcode Curl_multi_add_perform(struct Curl_multi *multi,
                                 struct Curl_easy *data,
                                 struct connectdata *conn)
{
  CURLMcode rc;

  if(multi->in_callback)
    return CURLM_RECURSIVE_API_CALL;

  rc = curl_multi_add_handle(multi, data);
  if(!rc) {
    struct SingleRequest *k = &data->req;

    /* pass in NULL for 'conn' here since we don't want to init the
       connection, only this transfer */
    Curl_init_do(data, NULL);

    /* take this handle to the perform state right away */
    multistate(data, CURLM_STATE_PERFORM);
    Curl_attach_connnection(data, conn);
    k->keepon |= KEEP_RECV; /* setup to receive! */
  }
  return rc;
}

/*
 * do_complete is called when the DO actions are complete.
 *
 * We init chunking and trailer bits to their default values here immediately
 * before receiving any header data for the current request.
 */
static void do_complete(struct connectdata *conn)
{
  conn->data->req.chunk = FALSE;
  Curl_pgrsTime(conn->data, TIMER_PRETRANSFER);
}

static CURLcode multi_do(struct Curl_easy *data, bool *done)
{
  CURLcode result = CURLE_OK;
  struct connectdata *conn = data->conn;

  DEBUGASSERT(conn);
  DEBUGASSERT(conn->handler);
  DEBUGASSERT(conn->data == data);

  if(conn->handler->do_it) {
    /* generic protocol-specific function pointer set in curl_connect() */
    result = conn->handler->do_it(conn, done);

    if(!result && *done)
      /* do_complete must be called after the protocol-specific DO function */
      do_complete(conn);
  }
  return result;
}

/*
 * multi_do_more() is called during the DO_MORE multi state. It is basically a
 * second stage DO state which (wrongly) was introduced to support FTP's
 * second connection.
 *
 * 'complete' can return 0 for incomplete, 1 for done and -1 for go back to
 * DOING state there's more work to do!
 */

static CURLcode multi_do_more(struct connectdata *conn, int *complete)
{
  CURLcode result = CURLE_OK;

  *complete = 0;

  if(conn->handler->do_more)
    result = conn->handler->do_more(conn, complete);

  if(!result && (*complete == 1))
    /* do_complete must be called after the protocol-specific DO function */
    do_complete(conn);

  return result;
}

/*
 * We are doing protocol-specific connecting and this is being called over and
 * over from the multi interface until the connection phase is done on
 * protocol layer.
 */

static CURLcode protocol_connecting(struct connectdata *conn,
                                    bool *done)
{
  CURLcode result = CURLE_OK;

  if(conn && conn->handler->connecting) {
    *done = FALSE;
    result = conn->handler->connecting(conn, done);
  }
  else
    *done = TRUE;

  return result;
}

/*
 * We are DOING this is being called over and over from the multi interface
 * until the DOING phase is done on protocol layer.
 */

static CURLcode protocol_doing(struct connectdata *conn, bool *done)
{
  CURLcode result = CURLE_OK;

  if(conn && conn->handler->doing) {
    *done = FALSE;
    result = conn->handler->doing(conn, done);
  }
  else
    *done = TRUE;

  return result;
}

/*
 * We have discovered that the TCP connection has been successful, we can now
 * proceed with some action.
 *
 */
static CURLcode protocol_connect(struct connectdata *conn,
                                 bool *protocol_done)
{
  CURLcode result = CURLE_OK;

  DEBUGASSERT(conn);
  DEBUGASSERT(protocol_done);

  *protocol_done = FALSE;

  if(conn->bits.tcpconnect[FIRSTSOCKET] && conn->bits.protoconnstart) {
    /* We already are connected, get back. This may happen when the connect
       worked fine in the first call, like when we connect to a local server
       or proxy. Note that we don't know if the protocol is actually done.

       Unless this protocol doesn't have any protocol-connect callback, as
       then we know we're done. */
    if(!conn->handler->connecting)
      *protocol_done = TRUE;

    return CURLE_OK;
  }

  if(!conn->bits.protoconnstart) {

    result = Curl_proxy_connect(conn, FIRSTSOCKET);
    if(result)
      return result;

    if(CONNECT_FIRSTSOCKET_PROXY_SSL())
      /* wait for HTTPS proxy SSL initialization to complete */
      return CURLE_OK;

    if(conn->bits.tunnel_proxy && conn->bits.httpproxy &&
       Curl_connect_ongoing(conn))
      /* when using an HTTP tunnel proxy, await complete tunnel establishment
         before proceeding further. Return CURLE_OK so we'll be called again */
      return CURLE_OK;

    if(conn->handler->connect_it) {
      /* is there a protocol-specific connect() procedure? */

      /* Call the protocol-specific connect function */
      result = conn->handler->connect_it(conn, protocol_done);
    }
    else
      *protocol_done = TRUE;

    /* it has started, possibly even completed but that knowledge isn't stored
       in this bit! */
    if(!result)
      conn->bits.protoconnstart = TRUE;
  }

  return result; /* pass back status */
}


static CURLMcode multi_runsingle(struct Curl_multi *multi,
                                 struct curltime now,
                                 struct Curl_easy *data)
{
  struct Curl_message *msg = NULL;
  bool connected;
  bool async;
  bool protocol_connected = FALSE;
  bool dophase_done = FALSE;
  bool done = FALSE;
  CURLMcode rc;
  CURLcode result = CURLE_OK;
  timediff_t timeout_ms;
  timediff_t recv_timeout_ms;
  timediff_t send_timeout_ms;
  int control;

  if(!GOOD_EASY_HANDLE(data))
    return CURLM_BAD_EASY_HANDLE;

  do {
    /* A "stream" here is a logical stream if the protocol can handle that
       (HTTP/2), or the full connection for older protocols */
    bool stream_error = FALSE;
    rc = CURLM_OK;

    DEBUGASSERT((data->mstate <= CURLM_STATE_CONNECT) ||
                (data->mstate >= CURLM_STATE_DONE) ||
                data->conn);
    if(!data->conn &&
       data->mstate > CURLM_STATE_CONNECT &&
       data->mstate < CURLM_STATE_DONE) {
      /* In all these states, the code will blindly access 'data->conn'
         so this is precaution that it isn't NULL. And it silences static
         analyzers. */
      failf(data, "In state %d with no conn, bail out!\n", data->mstate);
      return CURLM_INTERNAL_ERROR;
    }

    if(multi_ischanged(multi, TRUE)) {
      DEBUGF(infof(data, "multi changed, check CONNECT_PEND queue!\n"));
      process_pending_handles(multi); /* multiplexed */
    }

    if(data->conn && data->mstate > CURLM_STATE_CONNECT &&
       data->mstate < CURLM_STATE_COMPLETED) {
      /* Make sure we set the connection's current owner */
      data->conn->data = data;
    }

    if(data->conn &&
       (data->mstate >= CURLM_STATE_CONNECT) &&
       (data->mstate < CURLM_STATE_COMPLETED)) {
      /* we need to wait for the connect state as only then is the start time
         stored, but we must not check already completed handles */
      timeout_ms = Curl_timeleft(data, &now,
                                 (data->mstate <= CURLM_STATE_DO)?
                                 TRUE:FALSE);

      if(timeout_ms < 0) {
        /* Handle timed out */
        if(data->mstate == CURLM_STATE_WAITRESOLVE)
          failf(data, "Resolving timed out after %" CURL_FORMAT_TIMEDIFF_T
                " milliseconds",
                Curl_timediff(now, data->progress.t_startsingle));
        else if(data->mstate == CURLM_STATE_WAITCONNECT)
          failf(data, "Connection timed out after %" CURL_FORMAT_TIMEDIFF_T
                " milliseconds",
                Curl_timediff(now, data->progress.t_startsingle));
        else {
          struct SingleRequest *k = &data->req;
          if(k->size != -1) {
            failf(data, "Operation timed out after %" CURL_FORMAT_TIMEDIFF_T
                  " milliseconds with %" CURL_FORMAT_CURL_OFF_T " out of %"
                  CURL_FORMAT_CURL_OFF_T " bytes received",
                  Curl_timediff(now, data->progress.t_startsingle),
                  k->bytecount, k->size);
          }
          else {
            failf(data, "Operation timed out after %" CURL_FORMAT_TIMEDIFF_T
                  " milliseconds with %" CURL_FORMAT_CURL_OFF_T
                  " bytes received",
                  Curl_timediff(now, data->progress.t_startsingle),
                  k->bytecount);
          }
        }

        /* Force connection closed if the connection has indeed been used */
        if(data->mstate > CURLM_STATE_DO) {
          streamclose(data->conn, "Disconnected with pending data");
          stream_error = TRUE;
        }
        result = CURLE_OPERATION_TIMEDOUT;
        (void)multi_done(data, result, TRUE);
        /* Skip the statemachine and go directly to error handling section. */
        goto statemachine_end;
      }
    }

    switch(data->mstate) {
    case CURLM_STATE_INIT:
      /* init this transfer. */
      result = Curl_pretransfer(data);

      if(!result) {
        /* after init, go CONNECT */
        multistate(data, CURLM_STATE_CONNECT);
        Curl_pgrsTime(data, TIMER_STARTOP);
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      break;

    case CURLM_STATE_CONNECT_PEND:
      /* We will stay here until there is a connection available. Then
         we try again in the CURLM_STATE_CONNECT state. */
      break;

    case CURLM_STATE_CONNECT:
      /* Connect. We want to get a connection identifier filled in. */
      Curl_pgrsTime(data, TIMER_STARTSINGLE);
      if(data->set.timeout)
        Curl_expire(data, data->set.timeout, EXPIRE_TIMEOUT);

      if(data->set.connecttimeout)
        Curl_expire(data, data->set.connecttimeout, EXPIRE_CONNECTTIMEOUT);

      result = Curl_connect(data, &async, &protocol_connected);
      if(CURLE_NO_CONNECTION_AVAILABLE == result) {
        /* There was no connection available. We will go to the pending
           state and wait for an available connection. */
        multistate(data, CURLM_STATE_CONNECT_PEND);

        /* add this handle to the list of connect-pending handles */
        Curl_llist_insert_next(&multi->pending, multi->pending.tail, data,
                               &data->connect_queue);
        result = CURLE_OK;
        break;
      }
      else if(data->state.previouslypending) {
        /* this transfer comes from the pending queue so try move another */
        infof(data, "Transfer was pending, now try another\n");
        process_pending_handles(data->multi);
      }

      if(!result) {
        if(async)
          /* We're now waiting for an asynchronous name lookup */
          multistate(data, CURLM_STATE_WAITRESOLVE);
        else {
          /* after the connect has been sent off, go WAITCONNECT unless the
             protocol connect is already done and we can go directly to
             WAITDO or DO! */
          rc = CURLM_CALL_MULTI_PERFORM;

          if(protocol_connected)
            multistate(data, CURLM_STATE_DO);
          else {
#ifndef CURL_DISABLE_HTTP
            if(Curl_connect_ongoing(data->conn))
              multistate(data, CURLM_STATE_WAITPROXYCONNECT);
            else
#endif
              multistate(data, CURLM_STATE_WAITCONNECT);
          }
        }
      }
      break;

    case CURLM_STATE_WAITRESOLVE:
      /* awaiting an asynch name resolve to complete */
    {
      struct Curl_dns_entry *dns = NULL;
      struct connectdata *conn = data->conn;
      const char *hostname;

      DEBUGASSERT(conn);
      if(conn->bits.httpproxy)
        hostname = conn->http_proxy.host.name;
      else if(conn->bits.conn_to_host)
        hostname = conn->conn_to_host.name;
      else
        hostname = conn->host.name;

      /* check if we have the name resolved by now */
      dns = Curl_fetch_addr(conn, hostname, (int)conn->port);

      if(dns) {
#ifdef CURLRES_ASYNCH
        conn->async.dns = dns;
        conn->async.done = TRUE;
#endif
        result = CURLE_OK;
        infof(data, "Hostname '%s' was found in DNS cache\n", hostname);
      }

      if(!dns)
        result = Curl_resolv_check(data->conn, &dns);

      /* Update sockets here, because the socket(s) may have been
         closed and the application thus needs to be told, even if it
         is likely that the same socket(s) will again be used further
         down.  If the name has not yet been resolved, it is likely
         that new sockets have been opened in an attempt to contact
         another resolver. */
      singlesocket(multi, data);

      if(dns) {
        /* Perform the next step in the connection phase, and then move on
           to the WAITCONNECT state */
        result = Curl_once_resolved(data->conn, &protocol_connected);

        if(result)
          /* if Curl_once_resolved() returns failure, the connection struct
             is already freed and gone */
          data->conn = NULL; /* no more connection */
        else {
          /* call again please so that we get the next socket setup */
          rc = CURLM_CALL_MULTI_PERFORM;
          if(protocol_connected)
            multistate(data, CURLM_STATE_DO);
          else {
#ifndef CURL_DISABLE_HTTP
            if(Curl_connect_ongoing(data->conn))
              multistate(data, CURLM_STATE_WAITPROXYCONNECT);
            else
#endif
              multistate(data, CURLM_STATE_WAITCONNECT);
          }
        }
      }

      if(result) {
        /* failure detected */
        stream_error = TRUE;
        break;
      }
    }
    break;

#ifndef CURL_DISABLE_HTTP
    case CURLM_STATE_WAITPROXYCONNECT:
      /* this is HTTP-specific, but sending CONNECT to a proxy is HTTP... */
      DEBUGASSERT(data->conn);
      result = Curl_http_connect(data->conn, &protocol_connected);

      if(data->conn->bits.proxy_connect_closed) {
        rc = CURLM_CALL_MULTI_PERFORM;
        /* connect back to proxy again */
        result = CURLE_OK;
        multi_done(data, CURLE_OK, FALSE);
        multistate(data, CURLM_STATE_CONNECT);
      }
      else if(!result) {
        if((data->conn->http_proxy.proxytype != CURLPROXY_HTTPS ||
           data->conn->bits.proxy_ssl_connected[FIRSTSOCKET]) &&
           Curl_connect_complete(data->conn)) {
          rc = CURLM_CALL_MULTI_PERFORM;
          /* initiate protocol connect phase */
          multistate(data, CURLM_STATE_SENDPROTOCONNECT);
        }
      }
      else if(result)
        stream_error = TRUE;
      break;
#endif

    case CURLM_STATE_WAITCONNECT:
      /* awaiting a completion of an asynch TCP connect */
      DEBUGASSERT(data->conn);
      result = Curl_is_connected(data->conn, FIRSTSOCKET, &connected);
      if(connected && !result) {
#ifndef CURL_DISABLE_HTTP
        if((data->conn->http_proxy.proxytype == CURLPROXY_HTTPS &&
            !data->conn->bits.proxy_ssl_connected[FIRSTSOCKET]) ||
           Curl_connect_ongoing(data->conn)) {
          multistate(data, CURLM_STATE_WAITPROXYCONNECT);
          break;
        }
#endif
        rc = CURLM_CALL_MULTI_PERFORM;
        multistate(data, data->conn->bits.tunnel_proxy?
                   CURLM_STATE_WAITPROXYCONNECT:
                   CURLM_STATE_SENDPROTOCONNECT);
      }
      else if(result) {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, TRUE);
        stream_error = TRUE;
        break;
      }
      break;

    case CURLM_STATE_SENDPROTOCONNECT:
      result = protocol_connect(data->conn, &protocol_connected);
      if(!result && !protocol_connected)
        /* switch to waiting state */
        multistate(data, CURLM_STATE_PROTOCONNECT);
      else if(!result) {
        /* protocol connect has completed, go WAITDO or DO */
        multistate(data, CURLM_STATE_DO);
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      else if(result) {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, TRUE);
        stream_error = TRUE;
      }
      break;

    case CURLM_STATE_PROTOCONNECT:
      /* protocol-specific connect phase */
      result = protocol_connecting(data->conn, &protocol_connected);
      if(!result && protocol_connected) {
        /* after the connect has completed, go WAITDO or DO */
        multistate(data, CURLM_STATE_DO);
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      else if(result) {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, TRUE);
        stream_error = TRUE;
      }
      break;

    case CURLM_STATE_DO:
      if(data->set.connect_only) {
        /* keep connection open for application to use the socket */
        connkeep(data->conn, "CONNECT_ONLY");
        multistate(data, CURLM_STATE_DONE);
        result = CURLE_OK;
        rc = CURLM_CALL_MULTI_PERFORM;
      }
      else {
        /* Perform the protocol's DO action */
        result = multi_do(data, &dophase_done);

        /* When multi_do() returns failure, data->conn might be NULL! */

        if(!result) {
          if(!dophase_done) {
#ifndef CURL_DISABLE_FTP
            /* some steps needed for wildcard matching */
            if(data->state.wildcardmatch) {
              struct WildcardData *wc = &data->wildcard;
              if(wc->state == CURLWC_DONE || wc->state == CURLWC_SKIP) {
                /* skip some states if it is important */
                multi_done(data, CURLE_OK, FALSE);
                multistate(data, CURLM_STATE_DONE);
                rc = CURLM_CALL_MULTI_PERFORM;
                break;
              }
            }
#endif
            /* DO was not completed in one function call, we must continue
               DOING... */
            multistate(data, CURLM_STATE_DOING);
            rc = CURLM_OK;
          }

          /* after DO, go DO_DONE... or DO_MORE */
          else if(data->conn->bits.do_more) {
            /* we're supposed to do more, but we need to sit down, relax
               and wait a little while first */
            multistate(data, CURLM_STATE_DO_MORE);
            rc = CURLM_OK;
          }
          else {
            /* we're done with the DO, now DO_DONE */
            multistate(data, CURLM_STATE_DO_DONE);
            rc = CURLM_CALL_MULTI_PERFORM;
          }
        }
        else if((CURLE_SEND_ERROR == result) &&
                data->conn->bits.reuse) {
          /*
           * In this situation, a connection that we were trying to use
           * may have unexpectedly died.  If possible, send the connection
           * back to the CONNECT phase so we can try again.
           */
          char *newurl = NULL;
          followtype follow = FOLLOW_NONE;
          CURLcode drc;

          drc = Curl_retry_request(data->conn, &newurl);
          if(drc) {
            /* a failure here pretty much implies an out of memory */
            result = drc;
            stream_error = TRUE;
          }

          Curl_posttransfer(data);
          drc = multi_done(data, result, FALSE);

          /* When set to retry the connection, we must to go back to
           * the CONNECT state */
          if(newurl) {
            if(!drc || (drc == CURLE_SEND_ERROR)) {
              follow = FOLLOW_RETRY;
              drc = Curl_follow(data, newurl, follow);
              if(!drc) {
                multistate(data, CURLM_STATE_CONNECT);
                rc = CURLM_CALL_MULTI_PERFORM;
                result = CURLE_OK;
              }
              else {
                /* Follow failed */
                result = drc;
              }
            }
            else {
              /* done didn't return OK or SEND_ERROR */
              result = drc;
            }
          }
          else {
            /* Have error handler disconnect conn if we can't retry */
            stream_error = TRUE;
          }
          free(newurl);
        }
        else {
          /* failure detected */
          Curl_posttransfer(data);
          if(data->conn)
            multi_done(data, result, FALSE);
          stream_error = TRUE;
        }
      }
      break;

    case CURLM_STATE_DOING:
      /* we continue DOING until the DO phase is complete */
      DEBUGASSERT(data->conn);
      result = protocol_doing(data->conn, &dophase_done);
      if(!result) {
        if(dophase_done) {
          /* after DO, go DO_DONE or DO_MORE */
          multistate(data, data->conn->bits.do_more?
                     CURLM_STATE_DO_MORE:
                     CURLM_STATE_DO_DONE);
          rc = CURLM_CALL_MULTI_PERFORM;
        } /* dophase_done */
      }
      else {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, FALSE);
        stream_error = TRUE;
      }
      break;

    case CURLM_STATE_DO_MORE:
      /*
       * When we are connected, DO MORE and then go DO_DONE
       */
      DEBUGASSERT(data->conn);
      result = multi_do_more(data->conn, &control);

      if(!result) {
        if(control) {
          /* if positive, advance to DO_DONE
             if negative, go back to DOING */
          multistate(data, control == 1?
                     CURLM_STATE_DO_DONE:
                     CURLM_STATE_DOING);
          rc = CURLM_CALL_MULTI_PERFORM;
        }
        else
          /* stay in DO_MORE */
          rc = CURLM_OK;
      }
      else {
        /* failure detected */
        Curl_posttransfer(data);
        multi_done(data, result, FALSE);
        stream_error = TRUE;
      }
      break;

    case CURLM_STATE_DO_DONE:
      DEBUGASSERT(data->conn);
      if(data->conn->bits.multiplex)
        /* Check if we can move pending requests to send pipe */
        process_pending_handles(multi); /*  multiplexed */

      /* Only perform the transfer if there's a good socket to work with.
         Having both BAD is a signal to skip immediately to DONE */
      if((data->conn->sockfd != CURL_SOCKET_BAD) ||
         (data->conn->writesockfd != CURL_SOCKET_BAD))
        multistate(data, CURLM_STATE_PERFORM);
      else {
#ifndef CURL_DISABLE_FTP
        if(data->state.wildcardmatch &&
           ((data->conn->handler->flags & PROTOPT_WILDCARD) == 0)) {
          data->wildcard.state = CURLWC_DONE;
        }
#endif
        multistate(data, CURLM_STATE_DONE);
      }
      rc = CURLM_CALL_MULTI_PERFORM;
      break;

    case CURLM_STATE_TOOFAST: /* limit-rate exceeded in either direction */
      DEBUGASSERT(data->conn);
      /* if both rates are within spec, resume transfer */
      if(Curl_pgrsUpdate(data->conn))
        result = CURLE_ABORTED_BY_CALLBACK;
      else
        result = Curl_speedcheck(data, now);

      if(!result) {
        send_timeout_ms = 0;
        if(data->set.max_send_speed > 0)
          send_timeout_ms =
            Curl_pgrsLimitWaitTime(data->progress.uploaded,
                                   data->progress.ul_limit_size,
                                   data->set.max_send_speed,
                                   data->progress.ul_limit_start,
                                   now);

        recv_timeout_ms = 0;
        if(data->set.max_recv_speed > 0)
          recv_timeout_ms =
            Curl_pgrsLimitWaitTime(data->progress.downloaded,
                                   data->progress.dl_limit_size,
                                   data->set.max_recv_speed,
                                   data->progress.dl_limit_start,
                                   now);

        if(!send_timeout_ms && !recv_timeout_ms) {
          multistate(data, CURLM_STATE_PERFORM);
          Curl_ratelimit(data, now);
        }
        else if(send_timeout_ms >= recv_timeout_ms)
          Curl_expire(data, send_timeout_ms, EXPIRE_TOOFAST);
        else
          Curl_expire(data, recv_timeout_ms, EXPIRE_TOOFAST);
      }
      break;

    case CURLM_STATE_PERFORM:
    {
      char *newurl = NULL;
      bool retry = FALSE;
      bool comeback = FALSE;

      /* check if over send speed */
      send_timeout_ms = 0;
      if(data->set.max_send_speed > 0)
        send_timeout_ms = Curl_pgrsLimitWaitTime(data->progress.uploaded,
                                                 data->progress.ul_limit_size,
                                                 data->set.max_send_speed,
                                                 data->progress.ul_limit_start,
                                                 now);

      /* check if over recv speed */
      recv_timeout_ms = 0;
      if(data->set.max_recv_speed > 0)
        recv_timeout_ms = Curl_pgrsLimitWaitTime(data->progress.downloaded,
                                                 data->progress.dl_limit_size,
                                                 data->set.max_recv_speed,
                                                 data->progress.dl_limit_start,
                                                 now);

      if(send_timeout_ms || recv_timeout_ms) {
        Curl_ratelimit(data, now);
