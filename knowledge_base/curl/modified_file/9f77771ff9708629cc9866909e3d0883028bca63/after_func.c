  fd_set fd;
  struct timeval interval;
  int rc;

  /* now select() until we get connect or timeout */
  FD_ZERO(&fd);
  FD_SET(sockfd, &fd);

  interval.tv_sec = timeout_msec/1000;
  timeout_msec -= interval.tv_sec*1000;

  interval.tv_usec = timeout_msec*1000;

  rc = select(sockfd+1, NULL, &fd, NULL, &interval);
  if(-1 == rc)
    /* error, no connect here, try next */
    return -1;
  
  else if(0 == rc)
    /* timeout, no connect today */
    return 1;

  /* we have a connect! */
  return 0;
}

/*
 * TCP connect to the given host with timeout, proxy or remote doesn't matter.
 * There might be more than one IP address to try out. Fill in the passed
 * pointer with the connected socket.
 */

CURLcode Curl_connecthost(struct connectdata *conn,
                          long timeout_ms,
                          Curl_addrinfo *remotehost,
                          int port,
                          int sockfd, /* input socket, or -1 if none */
                          int *socket)
{
  struct SessionHandle *data = conn->data;
  int rc;

  struct timeval after;
  struct timeval before = Curl_tvnow();

#ifdef ENABLE_IPV6
  struct addrinfo *ai;
  /*
   * Connecting with IPv6 support is so much easier and cleanly done
   */
  port =0; /* we already have port in the 'remotehost' struct */

  if(sockfd != -1)
    /* don't use any previous one, it might be of wrong type */
    sclose(sockfd);
  sockfd = -1; /* none! */
  for (ai = remotehost; ai; ai = ai->ai_next) {
    sockfd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (sockfd < 0)
      continue;

    /* set socket non-blocking */
    nonblock(sockfd, TRUE);

    rc = connect(sockfd, ai->ai_addr, ai->ai_addrlen);

    if(0 == rc)
      /* direct connect, awesome! */
      break;

    /* asynchronous connect, wait for connect or timeout */
    rc = waitconnect(sockfd, timeout_ms);
    if(0 != rc) {
      /* connect failed or timed out */
      sclose(sockfd);
      sockfd = -1;

      /* get a new timeout for next attempt */
      after = Curl_tvnow();
      timeout_ms -= (long)(Curl_tvdiff(after, before)*1000);
      if(timeout_ms < 0)
        break;
      before = after;
      continue;
    }

    /* now disable the non-blocking mode again */
    nonblock(sockfd, FALSE);
    break;
  }
  conn->ai = ai;
  if (sockfd < 0) {
    failf(data, strerror(errno));
    return CURLE_COULDNT_CONNECT;
  }
#else
  /*
   * Connecting with IPv4-only support
   */
  int aliasindex;
  struct sockaddr_in serv_addr;

  if(-1 == sockfd)
    /* create an ordinary socket if none was provided */
    sockfd = socket(AF_INET, SOCK_STREAM, 0);

  if(-1 == sockfd)
    return CURLE_COULDNT_CONNECT; /* big time error */

  /* non-block socket */
  nonblock(sockfd, TRUE);

  /* This is the loop that attempts to connect to all IP-addresses we
     know for the given host. One by one. */
  for(rc=-1, aliasindex=0;
      rc && (struct in_addr *)remotehost->h_addr_list[aliasindex];
      aliasindex++) {

    /* copy this particular name info to the conn struct as it might
       be used later in the krb4 "system" */
    memset((char *) &serv_addr, '\0', sizeof(serv_addr));
    memcpy((char *)&(serv_addr.sin_addr),
           (struct in_addr *)remotehost->h_addr_list[aliasindex],
           sizeof(struct in_addr));
    serv_addr.sin_family = remotehost->h_addrtype;
    serv_addr.sin_port = htons(port);
  
    rc = connect(sockfd, (struct sockaddr *)&serv_addr,
                 sizeof(serv_addr));

    if(-1 == rc) {
      int error;
#ifdef WIN32
      error = (int)GetLastError();
#else
      error = errno;
#endif
      switch (error) {
      case EINPROGRESS:
      case EWOULDBLOCK:
#if defined(EAGAIN) && EAGAIN != EWOULDBLOCK
        /* On some platforms EAGAIN and EWOULDBLOCK are the
         * same value, and on others they are different, hence
         * the odd #if
         */
      case EAGAIN:
#endif

        /* asynchronous connect, wait for connect or timeout */
