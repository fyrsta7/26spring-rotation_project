
static void (*before_sync_wait)(void)= 0;
static void (*after_sync_wait)(void)= 0;

void thr_set_sync_wait_callback(void (*before_wait)(void),
                                void (*after_wait)(void))
{
  before_sync_wait= before_wait;
  after_sync_wait= after_wait;
}

/*
  Sync data in file to disk

  SYNOPSIS
    my_sync()
    fd			File descritor to sync
    my_flags		Flags (now only MY_WME is supported)

  NOTE
    If file system supports its, only file data is synced, not inode data.

    MY_IGNORE_BADFD is useful when fd is "volatile" - not protected by a
    mutex. In this case by the time of fsync(), fd may be already closed by
    another thread, or even reassigned to a different file. With this flag -
    MY_IGNORE_BADFD - such a situation will not be considered an error.
    (which is correct behaviour, if we know that the other thread synced the
    file before closing)

  RETURN
    0 ok
    -1 error
*/

int my_sync(File fd, myf my_flags)
{
  int res;
  DBUG_ENTER("my_sync");
  DBUG_PRINT("my",("Fd: %d  my_flags: %d", fd, my_flags));

  if (before_sync_wait)
    (*before_sync_wait)();
  do
  {
#if defined(F_FULLFSYNC)
    /*
      In Mac OS X >= 10.3 this call is safer than fsync() (it forces the
      disk's cache and guarantees ordered writes).
    */
    if (!(res= fcntl(fd, F_FULLFSYNC, 0)))
      break; /* ok */
    /* Some file systems don't support F_FULLFSYNC and fail above: */
    DBUG_PRINT("info",("fcntl(F_FULLFSYNC) failed, falling back"));
#endif
#if defined(HAVE_FDATASYNC) && HAVE_DECL_FDATASYNC
    res= fdatasync(fd);
#elif defined(HAVE_FSYNC)
    res= fsync(fd);
#elif defined(_WIN32)
    res= my_win_fsync(fd);
