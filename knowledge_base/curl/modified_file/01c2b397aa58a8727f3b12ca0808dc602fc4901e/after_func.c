    return GETSOCK_READSOCK(0);

  return GETSOCK_WRITESOCK(0);
}

static int domore_getsock(struct connectdata *conn,
                          curl_socket_t *sock,
                          int numsocks)
{
  if(!numsocks)
    return GETSOCK_BLANK;

  /* When in DO_MORE state, we could be either waiting for us
     to connect to a remote site, or we could wait for that site
     to connect to us. It makes a difference in the way: if we
     connect to the site we wait for the socket to become writable, if
     the site connects to us we wait for it to become readable */
  sock[0] = conn->sock[SECONDARYSOCKET];

  return GETSOCK_WRITESOCK(0);
}

/* returns bitmapped flags for this handle and its sockets */
static int multi_getsock(struct Curl_one_easy *easy,
                         curl_socket_t *socks, /* points to numsocks number
                                                  of sockets */
                         int numsocks)
{
  /* If the pipe broke, or if there's no connection left for this easy handle,
     then we MUST bail out now with no bitmask set. The no connection case can
     happen when this is called from curl_multi_remove_handle() =>
     singlesocket() => multi_getsock().
  */
  if(easy->easy_handle->state.pipe_broke || !easy->easy_conn)
    return 0;

  if(easy->state > CURLM_STATE_CONNECT &&
     easy->state < CURLM_STATE_COMPLETED) {
    /* Set up ownership correctly */
    easy->easy_conn->data = easy->easy_handle;
  }

  switch(easy->state) {
  default:
#if 0 /* switch back on these cases to get the compiler to check for all enums
         to be present */
  case CURLM_STATE_TOOFAST:  /* returns 0, so will not select. */
  case CURLM_STATE_COMPLETED:
  case CURLM_STATE_INIT:
  case CURLM_STATE_CONNECT:
  case CURLM_STATE_WAITDO:
  case CURLM_STATE_DONE:
  case CURLM_STATE_LAST:
    /* this will get called with CURLM_STATE_COMPLETED when a handle is
       removed */
#endif
    return 0;

  case CURLM_STATE_WAITRESOLVE:
    return Curl_resolv_getsock(easy->easy_conn, socks, numsocks);

  case CURLM_STATE_PROTOCONNECT:
    return Curl_protocol_getsock(easy->easy_conn, socks, numsocks);

  case CURLM_STATE_DO:
  case CURLM_STATE_DOING:
    return Curl_doing_getsock(easy->easy_conn, socks, numsocks);

  case CURLM_STATE_WAITPROXYCONNECT:
  case CURLM_STATE_WAITCONNECT:
    return waitconnect_getsock(easy->easy_conn, socks, numsocks);

  case CURLM_STATE_DO_MORE:
    return domore_getsock(easy->easy_conn, socks, numsocks);

  case CURLM_STATE_DO_DONE: /* since is set after DO is completed, we switch
                               to waiting for the same as the *PERFORM states */
  case CURLM_STATE_PERFORM:
  case CURLM_STATE_WAITPERFORM:
    return Curl_single_getsock(easy->easy_conn, socks, numsocks);
  }

}

CURLMcode curl_multi_fdset(CURLM *multi_handle,
                           fd_set *read_fd_set, fd_set *write_fd_set,
                           fd_set *exc_fd_set, int *max_fd)
{
  /* Scan through all the easy handles to get the file descriptors set.
     Some easy handles may not have connected to the remote host yet,
     and then we must make sure that is done. */
  struct Curl_multi *multi=(struct Curl_multi *)multi_handle;
  struct Curl_one_easy *easy;
  int this_max_fd=-1;
  curl_socket_t sockbunch[MAX_SOCKSPEREASYHANDLE];
  int bitmap;
  int i;
  (void)exc_fd_set; /* not used */

  if(!GOOD_MULTI_HANDLE(multi))
    return CURLM_BAD_HANDLE;

  easy=multi->easy.next;
  while(easy != &multi->easy) {
    bitmap = multi_getsock(easy, sockbunch, MAX_SOCKSPEREASYHANDLE);

    for(i=0; i< MAX_SOCKSPEREASYHANDLE; i++) {
      curl_socket_t s = CURL_SOCKET_BAD;

      if(bitmap & GETSOCK_READSOCK(i)) {
        FD_SET(sockbunch[i], read_fd_set);
        s = sockbunch[i];
      }
      if(bitmap & GETSOCK_WRITESOCK(i)) {
        FD_SET(sockbunch[i], write_fd_set);
        s = sockbunch[i];
      }
      if(s == CURL_SOCKET_BAD)
        /* this socket is unused, break out of loop */
        break;
      else {
        if((int)s > this_max_fd)
          this_max_fd = (int)s;
      }
    }

    easy = easy->next; /* check next handle */
  }

  *max_fd = this_max_fd;

  return CURLM_OK;
}

static CURLMcode multi_runsingle(struct Curl_multi *multi,
                                 struct Curl_one_easy *easy)
{
  struct Curl_message *msg = NULL;
  bool connected;
  bool async;
  bool protocol_connect = FALSE;
  bool dophase_done;
  bool done = FALSE;
  CURLMcode result = CURLM_OK;
  struct SingleRequest *k;

  if(!GOOD_EASY_HANDLE(easy->easy_handle))
    return CURLM_BAD_EASY_HANDLE;

  do {
    /* this is a do-while loop just to allow a break to skip to the end
       of it */
    bool disconnect_conn = FALSE;

    /* Handle the case when the pipe breaks, i.e., the connection
       we're using gets cleaned up and we're left with nothing. */
    if(easy->easy_handle->state.pipe_broke) {
      infof(easy->easy_handle, "Pipe broke: handle 0x%p, url = %s\n",
            easy, easy->easy_handle->state.path);

      if(easy->state != CURLM_STATE_COMPLETED) {
        /* Head back to the CONNECT state */
        multistate(easy, CURLM_STATE_CONNECT);
        result = CURLM_CALL_MULTI_PERFORM;
        easy->result = CURLE_OK;
      }

      easy->easy_handle->state.pipe_broke = FALSE;
      easy->easy_conn = NULL;
      break;
    }

    if(easy->easy_conn && easy->state > CURLM_STATE_CONNECT &&
       easy->state < CURLM_STATE_COMPLETED)
      /* Make sure we set the connection's current owner */
      easy->easy_conn->data = easy->easy_handle;

    switch(easy->state) {
    case CURLM_STATE_INIT:
      /* init this transfer. */
      easy->result=Curl_pretransfer(easy->easy_handle);

      if(CURLE_OK == easy->result) {
        /* after init, go CONNECT */
        multistate(easy, CURLM_STATE_CONNECT);
        result = CURLM_CALL_MULTI_PERFORM;

        easy->easy_handle->state.used_interface = Curl_if_multi;
      }
      break;

    case CURLM_STATE_CONNECT:
      /* Connect. We get a connection identifier filled in. */
      Curl_pgrsTime(easy->easy_handle, TIMER_STARTSINGLE);
      easy->result = Curl_connect(easy->easy_handle, &easy->easy_conn,
                                  &async, &protocol_connect);

      if(CURLE_OK == easy->result) {
        /* Add this handle to the send or pend pipeline */
        easy->result = addHandleToSendOrPendPipeline(easy->easy_handle,
                                                     easy->easy_conn);
        if(CURLE_OK == easy->result) {
          if(async)
            /* We're now waiting for an asynchronous name lookup */
            multistate(easy, CURLM_STATE_WAITRESOLVE);
          else {
            /* after the connect has been sent off, go WAITCONNECT unless the
               protocol connect is already done and we can go directly to
               WAITDO or DO! */
            result = CURLM_CALL_MULTI_PERFORM;

            if(protocol_connect)
              multistate(easy, multi->pipelining_enabled?
                         CURLM_STATE_WAITDO:CURLM_STATE_DO);
            else {
#ifndef CURL_DISABLE_HTTP
              if(easy->easy_conn->bits.tunnel_connecting)
                multistate(easy, CURLM_STATE_WAITPROXYCONNECT);
              else
#endif
                multistate(easy, CURLM_STATE_WAITCONNECT);
            }
          }
        }
      }
      break;

    case CURLM_STATE_WAITRESOLVE:
      /* awaiting an asynch name resolve to complete */
    {
      struct Curl_dns_entry *dns = NULL;

      /* check if we have the name resolved by now */
      easy->result = Curl_is_resolved(easy->easy_conn, &dns);

      if(dns) {
        /* Update sockets here. Mainly because the socket(s) may have been
           closed and the application thus needs to be told, even if it is
           likely that the same socket(s) will again be used further down. */
        singlesocket(multi, easy);

        /* Perform the next step in the connection phase, and then move on
           to the WAITCONNECT state */
        easy->result = Curl_async_resolved(easy->easy_conn,
                                           &protocol_connect);

        if(CURLE_OK != easy->result)
          /* if Curl_async_resolved() returns failure, the connection struct
             is already freed and gone */
          easy->easy_conn = NULL;           /* no more connection */
        else {
          /* call again please so that we get the next socket setup */
          result = CURLM_CALL_MULTI_PERFORM;
          if(protocol_connect)
            multistate(easy, multi->pipelining_enabled?
                       CURLM_STATE_WAITDO:CURLM_STATE_DO);
          else {
#ifndef CURL_DISABLE_HTTP
            if(easy->easy_conn->bits.tunnel_connecting)
              multistate(easy, CURLM_STATE_WAITPROXYCONNECT);
            else
#endif
              multistate(easy, CURLM_STATE_WAITCONNECT);
          }
        }
      }

      if(CURLE_OK != easy->result) {
        /* failure detected */
        disconnect_conn = TRUE;
        break;
      }
    }
    break;

#ifndef CURL_DISABLE_HTTP
    case CURLM_STATE_WAITPROXYCONNECT:
      /* this is HTTP-specific, but sending CONNECT to a proxy is HTTP... */
      easy->result = Curl_http_connect(easy->easy_conn, &protocol_connect);

      if(easy->easy_conn->bits.proxy_connect_closed) {
        /* reset the error buffer */
        if(easy->easy_handle->set.errorbuffer)
          easy->easy_handle->set.errorbuffer[0] = '\0';
        easy->easy_handle->state.errorbuf = FALSE;

        easy->result = CURLE_OK;
        result = CURLM_CALL_MULTI_PERFORM;
        multistate(easy, CURLM_STATE_CONNECT);
      }
      else if (CURLE_OK == easy->result) {
        if(!easy->easy_conn->bits.tunnel_connecting)
          multistate(easy, CURLM_STATE_WAITCONNECT);
      }
      break;
#endif

    case CURLM_STATE_WAITCONNECT:
      /* awaiting a completion of an asynch connect */
      easy->result = Curl_is_connected(easy->easy_conn,
                                       FIRSTSOCKET,
                                       &connected);
      if(connected) {
        /* see if we need to do any proxy magic first once we connected */
        easy->result = Curl_connected_proxy(easy->easy_conn);

        if(!easy->result)
          /* if everything is still fine we do the protocol-specific connect
             setup */
          easy->result = Curl_protocol_connect(easy->easy_conn,
                                               &protocol_connect);
      }

      if(CURLE_OK != easy->result) {
        /* failure detected */
        /* Just break, the cleaning up is handled all in one place */
        disconnect_conn = TRUE;
        break;
      }

      if(connected) {
        if(!protocol_connect) {
          /* We have a TCP connection, but 'protocol_connect' may be false
             and then we continue to 'STATE_PROTOCONNECT'. If protocol
             connect is TRUE, we move on to STATE_DO.
             BUT if we are using a proxy we must change to WAITPROXYCONNECT
          */
#ifndef CURL_DISABLE_HTTP
          if(easy->easy_conn->bits.tunnel_connecting)
            multistate(easy, CURLM_STATE_WAITPROXYCONNECT);
          else
#endif
            multistate(easy, CURLM_STATE_PROTOCONNECT);

        }
        else
          /* after the connect has completed, go WAITDO or DO */
          multistate(easy, multi->pipelining_enabled?
                     CURLM_STATE_WAITDO:CURLM_STATE_DO);

        result = CURLM_CALL_MULTI_PERFORM;
      }
      break;

    case CURLM_STATE_PROTOCONNECT:
      /* protocol-specific connect phase */
      easy->result = Curl_protocol_connecting(easy->easy_conn,
                                              &protocol_connect);
      if((easy->result == CURLE_OK) && protocol_connect) {
        /* after the connect has completed, go WAITDO or DO */
        multistate(easy, multi->pipelining_enabled?
                   CURLM_STATE_WAITDO:CURLM_STATE_DO);
        result = CURLM_CALL_MULTI_PERFORM;
      }
      else if(easy->result) {
        /* failure detected */
        Curl_posttransfer(easy->easy_handle);
        Curl_done(&easy->easy_conn, easy->result, FALSE);
        disconnect_conn = TRUE;
      }
      break;

    case CURLM_STATE_WAITDO:
      /* Wait for our turn to DO when we're pipelining requests */
#ifdef DEBUGBUILD
      infof(easy->easy_handle, "Conn %ld send pipe %zu inuse %d athead %d\n",
            easy->easy_conn->connectindex,
            easy->easy_conn->send_pipe->size,
            easy->easy_conn->writechannel_inuse?1:0,
            isHandleAtHead(easy->easy_handle,
                           easy->easy_conn->send_pipe)?1:0);
#endif
      if(!easy->easy_conn->writechannel_inuse &&
         isHandleAtHead(easy->easy_handle,
                        easy->easy_conn->send_pipe)) {
        /* Grab the channel */
        easy->easy_conn->writechannel_inuse = TRUE;
        multistate(easy, CURLM_STATE_DO);
        result = CURLM_CALL_MULTI_PERFORM;
      }
      break;

    case CURLM_STATE_DO:
      if(easy->easy_handle->set.connect_only) {
        /* keep connection open for application to use the socket */
        easy->easy_conn->bits.close = FALSE;
        multistate(easy, CURLM_STATE_DONE);
        easy->result = CURLE_OK;
        result = CURLM_OK;
      }
      else {
        /* Perform the protocol's DO action */
        easy->result = Curl_do(&easy->easy_conn,
                               &dophase_done);

        if(CURLE_OK == easy->result) {
          if(!dophase_done) {
            /* some steps needed for wildcard matching */
            if(easy->easy_handle->set.wildcardmatch) {
              struct WildcardData *wc = &easy->easy_handle->wildcard;
              if(wc->state == CURLWC_DONE || wc->state == CURLWC_SKIP) {
                /* skip some states if it is important */
                Curl_done(&easy->easy_conn, CURLE_OK, FALSE);
                multistate(easy, CURLM_STATE_DONE);
                result = CURLM_CALL_MULTI_PERFORM;
                break;
              }
            }
            /* DO was not completed in one function call, we must continue
               DOING... */
            multistate(easy, CURLM_STATE_DOING);
            result = CURLM_OK;
          }

          /* after DO, go DO_DONE... or DO_MORE */
          else if(easy->easy_conn->bits.do_more) {
            /* we're supposed to do more, but we need to sit down, relax
               and wait a little while first */
            multistate(easy, CURLM_STATE_DO_MORE);
            result = CURLM_OK;
          }
          else {
            /* we're done with the DO, now DO_DONE */
            multistate(easy, CURLM_STATE_DO_DONE);
            result = CURLM_CALL_MULTI_PERFORM;
          }
        }
        else if ((CURLE_SEND_ERROR == easy->result) &&
                 easy->easy_conn->bits.reuse) {
          /*
           * In this situation, a connection that we were trying to use
           * may have unexpectedly died.  If possible, send the connection
           * back to the CONNECT phase so we can try again.
           */
          char *newurl = NULL;
          followtype follow=FOLLOW_NONE;
          CURLcode drc;
          bool retry = FALSE;

          drc = Curl_retry_request(easy->easy_conn, &newurl);
          if(drc) {
            /* a failure here pretty much implies an out of memory */
            easy->result = drc;
            disconnect_conn = TRUE;
          }
          else
            retry = (bool)(newurl?TRUE:FALSE);

          Curl_posttransfer(easy->easy_handle);
          drc = Curl_done(&easy->easy_conn, easy->result, FALSE);

          /* When set to retry the connection, we must to go back to
           * the CONNECT state */
          if(retry) {
            if ((drc == CURLE_OK) || (drc == CURLE_SEND_ERROR)) {
              follow = FOLLOW_RETRY;
              drc = Curl_follow(easy->easy_handle, newurl, follow);
              if(drc == CURLE_OK) {
                multistate(easy, CURLM_STATE_CONNECT);
                result = CURLM_CALL_MULTI_PERFORM;
                easy->result = CURLE_OK;
              }
              else {
                /* Follow failed */
                easy->result = drc;
                free(newurl);
              }
            }
            else {
              /* done didn't return OK or SEND_ERROR */
              easy->result = drc;
              free(newurl);
            }
          }
          else {
            /* Have error handler disconnect conn if we can't retry */
            disconnect_conn = TRUE;
          }
        }
        else {
          /* failure detected */
          Curl_posttransfer(easy->easy_handle);
          Curl_done(&easy->easy_conn, easy->result, FALSE);
          disconnect_conn = TRUE;
        }
      }
      break;

    case CURLM_STATE_DOING:
      /* we continue DOING until the DO phase is complete */
      easy->result = Curl_protocol_doing(easy->easy_conn,
                                         &dophase_done);
      if(CURLE_OK == easy->result) {
        if(dophase_done) {
          /* after DO, go PERFORM... or DO_MORE */
          if(easy->easy_conn->bits.do_more) {
            /* we're supposed to do more, but we need to sit down, relax
               and wait a little while first */
            multistate(easy, CURLM_STATE_DO_MORE);
            result = CURLM_OK;
          }
          else {
            /* we're done with the DO, now DO_DONE */
            multistate(easy, CURLM_STATE_DO_DONE);
            result = CURLM_CALL_MULTI_PERFORM;
          }
        } /* dophase_done */
      }
      else {
        /* failure detected */
        Curl_posttransfer(easy->easy_handle);
        Curl_done(&easy->easy_conn, easy->result, FALSE);
        disconnect_conn = TRUE;
      }
      break;

    case CURLM_STATE_DO_MORE:
      /* Ready to do more? */
      easy->result = Curl_is_connected(easy->easy_conn,
                                       SECONDARYSOCKET,
                                       &connected);
      if(connected) {
        /*
         * When we are connected, DO MORE and then go DO_DONE
         */
        easy->result = Curl_do_more(easy->easy_conn);

        /* No need to remove ourselves from the send pipeline here since that
           is done for us in Curl_done() */

        if(CURLE_OK == easy->result) {
          multistate(easy, CURLM_STATE_DO_DONE);
          result = CURLM_CALL_MULTI_PERFORM;
        }
        else {
          /* failure detected */
          Curl_posttransfer(easy->easy_handle);
          Curl_done(&easy->easy_conn, easy->result, FALSE);
          disconnect_conn = TRUE;
        }
      }
      break;

    case CURLM_STATE_DO_DONE:
      /* Move ourselves from the send to recv pipeline */
      moveHandleFromSendToRecvPipeline(easy->easy_handle, easy->easy_conn);
      /* Check if we can move pending requests to send pipe */
      checkPendPipeline(easy->easy_conn);
      multistate(easy, CURLM_STATE_WAITPERFORM);
      result = CURLM_CALL_MULTI_PERFORM;
      break;

    case CURLM_STATE_WAITPERFORM:
      /* Wait for our turn to PERFORM */
      if(!easy->easy_conn->readchannel_inuse &&
         isHandleAtHead(easy->easy_handle,
                        easy->easy_conn->recv_pipe)) {
        /* Grab the channel */
        easy->easy_conn->readchannel_inuse = TRUE;
        multistate(easy, CURLM_STATE_PERFORM);
        result = CURLM_CALL_MULTI_PERFORM;
      }
#ifdef DEBUGBUILD
      else {
        infof(easy->easy_handle, "Conn %ld recv pipe %zu inuse %d athead %d\n",
              easy->easy_conn->connectindex,
              easy->easy_conn->recv_pipe->size,
              easy->easy_conn->readchannel_inuse?1:0,
              isHandleAtHead(easy->easy_handle,
                             easy->easy_conn->recv_pipe)?1:0);
      }
#endif
      break;

    case CURLM_STATE_TOOFAST: /* limit-rate exceeded in either direction */
      /* if both rates are within spec, resume transfer */
      if( ( ( easy->easy_handle->set.max_send_speed == 0 ) ||
            ( easy->easy_handle->progress.ulspeed <
              easy->easy_handle->set.max_send_speed ) )  &&
          ( ( easy->easy_handle->set.max_recv_speed == 0 ) ||
            ( easy->easy_handle->progress.dlspeed <
              easy->easy_handle->set.max_recv_speed ) )
        )
        multistate(easy, CURLM_STATE_PERFORM);
      break;

    case CURLM_STATE_PERFORM:
      /* check if over speed */
      if( (  ( easy->easy_handle->set.max_send_speed > 0 ) &&
             ( easy->easy_handle->progress.ulspeed >
               easy->easy_handle->set.max_send_speed ) )  ||
          (  ( easy->easy_handle->set.max_recv_speed > 0 ) &&
             ( easy->easy_handle->progress.dlspeed >
               easy->easy_handle->set.max_recv_speed ) )
        ) {
        /* Transfer is over the speed limit. Change state.  TODO: Call
         * Curl_expire() with the time left until we're targeted to be below
         * the speed limit again. */
        multistate(easy, CURLM_STATE_TOOFAST );
        break;
      }

      /* read/write data if it is ready to do so */
      easy->result = Curl_readwrite(easy->easy_conn, &done);

      k = &easy->easy_handle->req;

      if(!(k->keepon & KEEP_RECV)) {
        /* We're done receiving */
        easy->easy_conn->readchannel_inuse = FALSE;
      }

      if(!(k->keepon & KEEP_SEND)) {
        /* We're done sending */
        easy->easy_conn->writechannel_inuse = FALSE;
      }

      if(easy->result) {
        /* The transfer phase returned error, we mark the connection to get
         * closed to prevent being re-used. This is because we can't possibly
         * know if the connection is in a good shape or not now.  Unless it is
         * a protocol which uses two "channels" like FTP, as then the error
         * happened in the data connection.
         */
        if(!(easy->easy_conn->protocol & PROT_DUALCHANNEL))
          easy->easy_conn->bits.close = TRUE;

        Curl_posttransfer(easy->easy_handle);
        Curl_done(&easy->easy_conn, easy->result, FALSE);
      }
      else if(TRUE == done) {
        char *newurl = NULL;
        bool retry = FALSE;
        followtype follow=FOLLOW_NONE;

        easy->result = Curl_retry_request(easy->easy_conn, &newurl);
        if(!easy->result)
          retry = (bool)(newurl?TRUE:FALSE);

