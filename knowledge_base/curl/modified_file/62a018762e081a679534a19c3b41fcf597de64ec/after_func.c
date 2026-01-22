
    if(wc < 0) {
      error = errno;
      if((error == EINTR) || (error == EAGAIN))
        continue;
      logmsg("writing to file descriptor: %d,", filedes);
      logmsg("unrecoverable write() failure: (%d) %s",
             error, strerror(error));
      return -1;
    }

    if(wc == 0) {
      logmsg("put 0 writing to stdout");
      return 0;
    }

    nwrite += wc;

  } while((size_t)nwrite < nbytes);

  if(verbose)
    logmsg("wrote %zd bytes", nwrite);

  return nwrite;
}

/*
 * read_stdin tries to read from stdin nbytes into the given buffer. This is a
 * blocking function that will only return TRUE when nbytes have actually been
 * read or FALSE when an unrecoverable error has been detected. Failure of this
 * function is an indication that the sockfilt process should terminate.
 */

static bool read_stdin(void *buffer, size_t nbytes)
{
  ssize_t nread = fullread(fileno(stdin), buffer, nbytes);
  if(nread != (ssize_t)nbytes) {
    logmsg("exiting...");
    return FALSE;
  }
  return TRUE;
}

/*
 * write_stdout tries to write to stdio nbytes from the given buffer. This is a
 * blocking function that will only return TRUE when nbytes have actually been
 * written or FALSE when an unrecoverable error has been detected. Failure of
 * this function is an indication that the sockfilt process should terminate.
 */

static bool write_stdout(const void *buffer, size_t nbytes)
{
  ssize_t nwrite = fullwrite(fileno(stdout), buffer, nbytes);
  if(nwrite != (ssize_t)nbytes) {
    logmsg("exiting...");
    return FALSE;
  }
  return TRUE;
}

static void lograw(unsigned char *buffer, ssize_t len)
{
  char data[120];
  ssize_t i;
  unsigned char *ptr = buffer;
  char *optr = data;
  ssize_t width=0;

  for(i=0; i<len; i++) {
    switch(ptr[i]) {
    case '\n':
      sprintf(optr, "\\n");
      width += 2;
      optr += 2;
      break;
    case '\r':
      sprintf(optr, "\\r");
      width += 2;
      optr += 2;
      break;
    default:
      sprintf(optr, "%c", (ISGRAPH(ptr[i]) || ptr[i]==0x20) ?ptr[i]:'.');
      width++;
      optr++;
      break;
    }

    if(width>60) {
      logmsg("'%s'", data);
      width = 0;
      optr = data;
    }
  }
  if(width)
    logmsg("'%s'", data);
}

#ifdef USE_WINSOCK
/*
 * WinSock select() does not support standard file descriptors,
 * it can only check SOCKETs. The following function is an attempt
 * to re-create a select() function with support for other handle types.
 *
 * select() function with support for WINSOCK2 sockets and all
 * other handle types supported by WaitForMultipleObjectsEx() as
 * well as disk files, anonymous and names pipes, and character input.
 *
