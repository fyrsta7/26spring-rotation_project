    goto out;
#endif

  sockfd = socket(domain, type, protocol);

  if (sockfd == -1)
    goto out;

  if (uv__nonblock(sockfd, 1) || uv__cloexec(sockfd, 1)) {
    close(sockfd);
    sockfd = -1;
  }

out:
  return sockfd;
}


int uv__accept(int sockfd) {
  int peerfd;

  assert(sockfd >= 0);

  while (1) {
#if __linux__
    peerfd = uv__accept4(sockfd,
                         NULL,
                         NULL,
                         UV__SOCK_NONBLOCK|UV__SOCK_CLOEXEC);

    if (peerfd != -1)
      break;

    if (errno == EINTR)
      continue;

    if (errno != ENOSYS)
      break;
#endif

    peerfd = accept(sockfd, NULL, NULL);
